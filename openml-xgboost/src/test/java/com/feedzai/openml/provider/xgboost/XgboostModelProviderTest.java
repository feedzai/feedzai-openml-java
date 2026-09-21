/*
 * Copyright 2026 Feedzai
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package com.feedzai.openml.provider.xgboost;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import ml.dmlc.xgboost4j.java.DMatrix;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

/**
 * Round-trip tests for the XGBoost provider: train (in-process, no Spark) -> export -> load -> score.
 *
 * <p>These tests exercise the native {@code xgboost4j} library, validating that the provider works
 * end-to-end on the host architecture (including ARM / Apple Silicon).
 *
 * @since 1.0.0
 */
public class XgboostModelProviderTest {

    /**
     * Binary target nominal values.
     */
    private static final Set<String> TARGET_VALUES = ImmutableSet.of("false", "true");

    /**
     * Number of predictive features in the test schema.
     */
    private static final int NUM_FEATURES = 4;

    /**
     * Schema used across the tests (4 numeric features + binary categorical target).
     */
    private static DatasetSchema schema;

    /**
     * Whether the native {@code xgboost4j} library can be loaded on this platform. Tests that train or
     * score are skipped when it cannot (e.g. musl/Alpine, for which XGBoost ships no native library).
     */
    private static boolean nativeAvailable;

    /**
     * Sets up the shared schema and detects native library availability.
     */
    @BeforeClass
    public static void setUp() {
        schema = MockDataset.generateDefaultSchema(TARGET_VALUES, NUM_FEATURES);
        nativeAvailable = xgboostNativeAvailable();
    }

    /**
     * Probes whether the native XGBoost library can be loaded on the current platform.
     *
     * <p>The published {@code xgboost4j} jar bundles glibc Linux (x86_64, aarch64), macOS (x86_64,
     * Apple Silicon) and Windows natives, but no musl build - so on Alpine/musl the load fails.
     *
     * @return {@code true} if the native library initializes successfully.
     */
    private static boolean xgboostNativeAvailable() {
        try {
            new DMatrix(new float[]{0f}, 1, 1, Float.NaN).dispose();
            return true;
        } catch (final Throwable t) {
            // UnsatisfiedLinkError / NoClassDefFoundError / XGBoostError: native unsupported here.
            return false;
        }
    }

    /**
     * Valid training parameters.
     *
     * @return The parameters map.
     */
    private static Map<String, String> trainParams() {
        return ImmutableMap.of(
                XgboostDescriptorUtil.OBJECTIVE_PARAMETER_NAME, "binary:logistic",
                XgboostDescriptorUtil.NUM_ROUND_PARAMETER_NAME, "10",
                "max_depth", "3",
                "eta", "0.3",
                XgboostDescriptorUtil.NTHREAD_PARAMETER_NAME, "1"
        );
    }

    /**
     * The provider exposes the XGBoost algorithm and resolves its creator by name.
     */
    @Test
    public void providerExposesXgboostAlgorithm() {
        final XgboostModelProvider provider = new XgboostModelProvider();

        assertThat(provider.getName()).isEqualTo("XGBoost");
        assertThat(provider.getAlgorithms())
                .extracting(MLAlgorithmDescriptor::getAlgorithmName)
                .contains("XGBoost Binary Classifier");
        assertThat(provider.getModelCreator("XGBoost Binary Classifier")).isPresent();
        assertThat(provider.getModelCreator("Non Existing Algorithm")).isEmpty();
    }

    /**
     * Valid fit parameters produce no validation errors.
     *
     * @throws Exception If the temporary directory cannot be created.
     */
    @Test
    public void validateForFitAcceptsValidParams() throws Exception {
        final Path tmpDir = Files.createTempDirectory("xgb_fit_validation_");
        final List<ParamValidationError> errors =
                new XgboostModelCreator().validateForFit(tmpDir, schema, trainParams());

        assertThat(errors).isEmpty();
    }

    /**
     * Trains a model in-process, then scores an instance: the class distribution must be a valid
     * probability distribution and {@code classify} must return the arg-max class.
     *
     * @throws Exception If training/scoring fails.
     */
    @Test
    public void trainsAndScoresInProcess() throws Exception {
        Assume.assumeTrue("XGBoost native library unavailable on this platform (e.g. musl/Alpine).", nativeAvailable);

        final XgboostModelCreator creator = new XgboostModelCreator();
        final Dataset trainDataset = new MockDataset(schema, 200, new Random(0));

        final XgboostClassificationModel model = creator.fit(trainDataset, new Random(0), trainParams());

        final MockInstance instance = new MockInstance(schema, new Random(7));
        final double[] distribution = model.getClassDistribution(instance);

        assertThat(distribution).hasSize(TARGET_VALUES.size());
        assertThat(distribution[0] + distribution[1]).isCloseTo(1.0, offset(1e-6));
        assertThat(distribution[0]).isBetween(0.0, 1.0);
        assertThat(distribution[1]).isBetween(0.0, 1.0);

        final int classIndex = model.classify(instance);
        assertThat(classIndex).isBetween(0, 1);
        assertThat(distribution[classIndex]).isGreaterThanOrEqualTo(distribution[1 - classIndex]);

        model.close();
    }

    /**
     * A model saved to disk and reloaded produces identical scores (export -> load round-trip).
     *
     * @throws Exception If training/saving/loading fails.
     */
    @Test
    public void savedModelReloadsWithIdenticalScores() throws Exception {
        Assume.assumeTrue("XGBoost native library unavailable on this platform (e.g. musl/Alpine).", nativeAvailable);

        final XgboostModelCreator creator = new XgboostModelCreator();
        final Dataset trainDataset = new MockDataset(schema, 200, new Random(1));

        final XgboostClassificationModel trainedModel = creator.fit(trainDataset, new Random(1), trainParams());

        final MockInstance instance = new MockInstance(schema, new Random(11));
        final double[] originalDistribution = trainedModel.getClassDistribution(instance);

        final Path saveDir = Files.createTempDirectory("xgb_save_");
        assertThat(trainedModel.save(saveDir, "reloaded")).isTrue();
        trainedModel.close();

        final XgboostClassificationModel reloadedModel = creator.loadModel(saveDir, schema);
        final double[] reloadedDistribution = reloadedModel.getClassDistribution(instance);

        assertThat(reloadedDistribution).containsExactly(originalDistribution, offset(1e-9));

        reloadedModel.close();
    }

    /**
     * Training on an empty dataset raises a {@link ModelTrainingException}.
     */
    @Test
    public void trainingOnEmptyDatasetThrows() {
        final XgboostModelCreator creator = new XgboostModelCreator();
        final Dataset emptyDataset = new MockDataset(schema, 0, new Random(0));

        assertThatThrownBy(() -> creator.fit(emptyDataset, new Random(0), trainParams()))
                .isInstanceOf(ModelTrainingException.class)
                .hasMessageContaining("empty");
    }

    /**
     * {@code validateForLoad} runs its validations and reports an error when no model exists in the
     * given directory. This path does not touch the native library.
     *
     * @throws Exception If the temporary directory cannot be created.
     */
    @Test
    public void validateForLoadReportsMissingModel() throws Exception {
        final Path emptyDir = Files.createTempDirectory("xgb_no_model_");

        final List<ParamValidationError> errors =
                new XgboostModelCreator().validateForLoad(emptyDir, schema, ImmutableMap.of());

        assertThat(errors).isNotEmpty();
    }

    /**
     * Loading an invalid (non-XGBoost) model file raises a {@link ModelLoadingException}.
     *
     * @throws Exception If file operations fail.
     */
    @Test
    public void loadModelThrowsOnInvalidModelFile() throws Exception {
        Assume.assumeTrue("XGBoost native library unavailable on this platform (e.g. musl/Alpine).", nativeAvailable);

        final Path root = Files.createTempDirectory("xgb_bad_model_");
        final Path modelDir = Files.createDirectory(root.resolve("model"));
        Files.write(modelDir.resolve(XgboostModelCreator.MODEL_BINARY_RESOURCE_FILE_NAME),
                "this is not a valid xgboost model".getBytes(StandardCharsets.UTF_8));

        assertThatThrownBy(() -> new XgboostModelCreator().loadModel(root, schema))
                .isInstanceOf(ModelLoadingException.class);
    }

    /**
     * {@link XgboostClassificationModel#save(Path, String)} returns {@code false} when persistence
     * fails (e.g. an unwritable target path).
     *
     * @throws Exception If training fails.
     */
    @Test
    public void saveReturnsFalseWhenPersistenceFails() throws Exception {
        Assume.assumeTrue("XGBoost native library unavailable on this platform (e.g. musl/Alpine).", nativeAvailable);

        final XgboostModelCreator creator = new XgboostModelCreator();
        final XgboostClassificationModel model =
                creator.fit(new MockDataset(schema, 50, new Random(0)), new Random(0), trainParams());

        final boolean saved = model.save(Paths.get("/this/path/does/not/exist/xyz"), "model");

        assertThat(saved).isFalse();
        model.close();
    }
}
