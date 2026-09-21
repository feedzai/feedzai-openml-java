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
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.mocks.MockInstance;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

/**
 * Unit tests for the pure-logic helpers of the XGBoost provider that do not require the native library,
 * so they run on every platform (including musl/Alpine where the native library is unavailable).
 *
 * @since 1.0.0
 */
public class XgboostInternalsTest {

    /**
     * Schema with the target in the middle position (index 1) and two numeric features.
     */
    private static final DatasetSchema SCHEMA_TARGET_IN_MIDDLE = new DatasetSchema(1, ImmutableList.of(
            new FieldSchema("f0", 0, new NumericValueSchema(false)),
            new FieldSchema("target", 1, new CategoricalValueSchema(false, ImmutableSet.of("0", "1"))),
            new FieldSchema("f2", 2, new NumericValueSchema(false))
    ));

    // --- XgboostSchemaUtils ---

    /**
     * The feature vector is built in field order, skipping the target column.
     */
    @Test
    public void featureRowSkipsTargetColumn() {
        assertThat(XgboostSchemaUtils.numFeatures(SCHEMA_TARGET_IN_MIDDLE)).isEqualTo(2);

        final MockInstance instance = new MockInstance(ImmutableList.of(10.0, 0.0, 30.0));
        final float[] row = XgboostSchemaUtils.featureRow(instance, SCHEMA_TARGET_IN_MIDDLE);

        assertThat(row).containsExactly(10.0f, 30.0f);
    }

    // --- XgboostModelCreator.numRoundOf ---

    /**
     * The number of rounds falls back to the default when absent or empty, and is parsed otherwise.
     */
    @Test
    public void numRoundOfHandlesDefaultAndValue() {
        assertThat(XgboostModelCreator.numRoundOf(ImmutableMap.of())).isEqualTo(100);
        assertThat(XgboostModelCreator.numRoundOf(ImmutableMap.of("num_round", ""))).isEqualTo(100);
        assertThat(XgboostModelCreator.numRoundOf(ImmutableMap.of("num_round", "50"))).isEqualTo(50);
    }

    // --- XgboostModelCreator.toBoosterParams ---

    /**
     * {@code num_round} and empty values are excluded, and objective/seed defaults are injected.
     */
    @Test
    public void toBoosterParamsExcludesNumRoundAndInjectsDefaults() {
        final Map<String, String> params = ImmutableMap.of(
                "num_round", "10",
                "eta", "0.3",
                "max_depth", ""
        );

        final Map<String, Object> boosterParams = XgboostModelCreator.toBoosterParams(params, new Random(0));

        assertThat(boosterParams).doesNotContainKey("num_round");
        assertThat(boosterParams).doesNotContainKey("max_depth");
        assertThat(boosterParams).containsEntry("eta", "0.3");
        assertThat(boosterParams).containsEntry("objective", "binary:logistic");
        assertThat(boosterParams).containsKey("seed");
    }

    /**
     * Explicit objective and seed are preserved (not overridden by defaults).
     */
    @Test
    public void toBoosterParamsPreservesExplicitValues() {
        final Map<String, String> params = ImmutableMap.of(
                "objective", "multi:softprob",
                "seed", "42"
        );

        final Map<String, Object> boosterParams = XgboostModelCreator.toBoosterParams(params, new Random(0));

        assertThat(boosterParams).containsEntry("objective", "multi:softprob");
        assertThat(boosterParams).containsEntry("seed", "42");
    }

    // --- XgboostModelCreator.resolveModelFile ---

    /**
     * A direct path to a file is returned unchanged.
     *
     * @throws Exception If file operations fail.
     */
    @Test
    public void resolveModelFileReturnsDirectFile() throws Exception {
        final Path file = Files.createTempFile("xgb_model_", ".ubj");
        assertThat(XgboostModelCreator.resolveModelFile(file)).isEqualTo(file);
    }

    /**
     * A directory with a {@code model/} sub-folder resolves to the file within it (Pulse layout).
     *
     * @throws Exception If file operations fail.
     */
    @Test
    public void resolveModelFileResolvesPulseModelFolderLayout() throws Exception {
        final Path root = Files.createTempDirectory("xgb_root_");
        final Path modelDir = Files.createDirectory(root.resolve("model"));
        final Path modelFile = Files.createFile(modelDir.resolve(XgboostModelCreator.MODEL_BINARY_RESOURCE_FILE_NAME));

        assertThat(XgboostModelCreator.resolveModelFile(root)).isEqualTo(modelFile);
    }

    /**
     * A directory without a {@code model/} sub-folder resolves to the model file at its root.
     *
     * @throws Exception If file operations fail.
     */
    @Test
    public void resolveModelFileResolvesRootLayout() throws Exception {
        final Path root = Files.createTempDirectory("xgb_root_flat_");

        assertThat(XgboostModelCreator.resolveModelFile(root))
                .isEqualTo(root.resolve(XgboostModelCreator.MODEL_BINARY_RESOURCE_FILE_NAME));
    }

    // --- XgboostClassificationModel.toClassDistribution ---

    /**
     * A single-value (binary) prediction expands to {@code [1 - p, p]}.
     */
    @Test
    public void toClassDistributionExpandsBinaryPrediction() {
        final double[] distribution = XgboostClassificationModel.toClassDistribution(new float[]{0.3f});

        assertThat(distribution).hasSize(2);
        assertThat(distribution[1]).isCloseTo(0.3f, offset(1e-6));
        assertThat(distribution[0]).isCloseTo(1.0 - 0.3f, offset(1e-6));
    }

    /**
     * A multi-value (multi-class) prediction is returned as-is.
     */
    @Test
    public void toClassDistributionPassesThroughMulticlassPrediction() {
        final double[] distribution = XgboostClassificationModel.toClassDistribution(new float[]{0.2f, 0.5f, 0.3f});

        assertThat(distribution).hasSize(3);
        assertThat(distribution[0]).isCloseTo(0.2f, offset(1e-6));
        assertThat(distribution[1]).isCloseTo(0.5f, offset(1e-6));
        assertThat(distribution[2]).isCloseTo(0.3f, offset(1e-6));
    }

    // --- misc no-native paths ---

    /**
     * {@link XgboostClassificationModel#getSchema()} returns the schema it was constructed with.
     */
    @Test
    public void getSchemaReturnsProvidedSchema() {
        final XgboostClassificationModel model = new XgboostClassificationModel(null, SCHEMA_TARGET_IN_MIDDLE);
        assertThat(model.getSchema()).isSameAs(SCHEMA_TARGET_IN_MIDDLE);
    }

    /**
     * Fitting with a schema that has no target field fails fast (before any native call).
     */
    @Test
    public void fitWithoutTargetSchemaThrows() {
        final DatasetSchema noTargetSchema = new DatasetSchema(ImmutableList.of(
                new FieldSchema("f0", 0, new NumericValueSchema(false)),
                new FieldSchema("f1", 1, new NumericValueSchema(false))
        ));
        final Dataset dataset = new MockDataset(noTargetSchema, 5, new Random(0));

        assertThatThrownBy(() -> new XgboostModelCreator().fit(dataset, new Random(0), ImmutableMap.of()))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("target");
    }
}
