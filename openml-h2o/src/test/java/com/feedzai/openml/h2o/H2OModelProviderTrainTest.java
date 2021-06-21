/*
 * Copyright 2018 Feedzai
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

package com.feedzai.openml.h2o;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.feedzai.openml.util.provider.AbstractProviderModelTrainTest;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;
import java.util.HashMap;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatCode;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;

/**
 * Tests for training models with {@link H2OModelProvider}.
 *
 * @author Luis Reis (luis.reis@feedzai.com)
 * @since 0.1.0
 */
public class H2OModelProviderTrainTest extends AbstractProviderModelTrainTest<AbstractClassificationH2OModel, H2OModelCreator, H2OModelProvider> implements H2ODatasetMixin {

    /**
     * Logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OModelProviderTrainTest.class);

    private Dataset dataset;

    /**
     * Creates the dataset to be used in these tests.
     *
     * <p>
     * Due to an Isolation Forest nuance, the dataset size must be bigger than the sample_size param, otherwise
     * Isolation forest will not be able to detect instances as anomalous.
     * </p>
     */
    @Before
    public void createDataset() {
        this.dataset = createDataset(SCHEMA);
    }

    /**
     * Creates a dataset to be used on the tests, based on the provided schema.
     *
     * @param schema The schema that the dataset will comply to.
     * @return A new dataset.
     */
    private Dataset createDataset(final DatasetSchema schema) {
        final Map<String, String> params = H2OAlgorithmTestParams.getIsolationForest();
        final int sampleSize = Optional.ofNullable(params.get("sample_size")).map(Integer::parseInt).orElse(256);

        return createDataset(schema, sampleSize + 100);
    }

    /**
     * Creates a dataset to be used on the tests, based on the provided schema and set.
     *
     * @param schema The schema that the dataset will comply to.
     * @param size   The number of instances of the dataset.
     * @return A new dataset.
     */
    private Dataset createDataset(final DatasetSchema schema, final int size) {
        final Random random = new Random(234);

        logger.info("Using dataset size of {}", size);
        final List<Instance> instances = IntStream.range(0, size)
                .mapToObj(index -> new MockInstance(schema, random))
                .collect(Collectors.toList());
        return new MockDataset(schema, instances);
    }

    @Override
    public AbstractClassificationH2OModel getFirstModel() throws ModelTrainingException {
        final H2OModelCreator modelCreator = getMachineLearningModelLoader(H2OAlgorithm.DEEP_LEARNING);
        return modelCreator.fit(TRAIN_DATASET, new Random(0), ImmutableMap.of());
    }

    @Override
    public AbstractClassificationH2OModel getSecondModel() throws ModelTrainingException {
        // Naive Bayes is used to exercise POJO logic as it is the only one that doesn't support MOJOs
        final H2OModelCreator modelCreator = getMachineLearningModelLoader(H2OAlgorithm.NAIVE_BAYES_CLASSIFIER);
        return modelCreator.fit(TRAIN_DATASET, new Random(1), ImmutableMap.of());
    }

    @Override
    public Set<Integer> getClassifyValuesOfFirstModel() {
        return IntStream.range(0, TARGET_VALUES.size()).boxed().collect(Collectors.toSet());
    }

    @Override
    public Set<Integer> getClassifyValuesOfSecondModel() {
        return IntStream.range(0, TARGET_VALUES.size()).boxed().collect(Collectors.toSet());
    }

    @Override
    public H2OModelCreator getFirstMachineLearningModelLoader() {
        return getMachineLearningModelLoader(H2OAlgorithm.DEEP_LEARNING);
    }

    @Override
    public H2OModelProvider getMachineLearningProvider() {
        return new H2OModelProvider();
    }

    @Override
    public Instance getDummyInstance() {
        return new MockInstance(createDatasetSchema(TARGET_VALUES), new Random(0));
    }

    @Override
    public Instance getDummyInstanceDifferentResult() {
        return new MockInstance(createDatasetSchema(TARGET_VALUES), new Random(1));
    }

    @Override
    public DatasetSchema createDatasetSchema(final Set<String> targetValues) {
        return SCHEMA;
    }

    @Override
    public MLAlgorithmEnum getValidAlgorithm() {
        return H2OAlgorithm.DEEP_LEARNING;
    }

    @Override
    public Set<String> getFirstModelTargetNominalValues() {
        return TARGET_VALUES;
    }

    @Override
    protected Dataset getTrainDataset() {
        return this.dataset;
    }

    @Override
    protected Map<MLAlgorithmEnum, Map<String, String>> getTrainAlgorithms() {
        final ImmutableMap.Builder<MLAlgorithmEnum, Map<String, String>> builder = new ImmutableMap.Builder<>();

        builder.put(H2OAlgorithm.DEEP_LEARNING, H2OAlgorithmTestParams.getDeepLearning());
        builder.put(H2OAlgorithm.DISTRIBUTED_RANDOM_FOREST, H2OAlgorithmTestParams.getDrf());
        builder.put(H2OAlgorithm.GRADIENT_BOOSTING_MACHINE, H2OAlgorithmTestParams.getGbm());
        builder.put(H2OAlgorithm.NAIVE_BAYES_CLASSIFIER, H2OAlgorithmTestParams.getBayes());
        builder.put(H2OAlgorithm.XG_BOOST, H2OAlgorithmTestParams.getXgboost());
        builder.put(H2OAlgorithm.GENERALIZED_LINEAR_MODEL, H2OAlgorithmTestParams.getGlm());
        builder.put(H2OAlgorithm.ISOLATION_FOREST, H2OAlgorithmTestParams.getIsolationForest());

        return builder.build();
    }

    /**
     * Tests that the model loaded by {@link H2OModelCreator#fit(Dataset, Random, Map)} is able to score instances,
     * based on a schema with no target variable.
     */
    @Test
    public final void testIsolationForestWithDatasetWithoutTargetVariable() throws ModelTrainingException {
        final H2OModelCreator loader = getMachineLearningModelLoader(H2OAlgorithm.ISOLATION_FOREST);
        final Map<String, String> params = H2OAlgorithmTestParams.getIsolationForest();

        final Random random = new Random(234);
        final Dataset dataset = createDataset(SCHEMA_NO_TARGET_VARIABLE);

        final AbstractClassificationH2OModel model = loader.fit(dataset, random, params);

        final MockInstance dummyInstance = new MockInstance(dataset.getSchema(), random);
        final double[] classDistribution = model.getClassDistribution(dummyInstance);
        assertThat(classDistribution).as("Scoring instance '%s' succeeds", dummyInstance)
                .hasSize(2)
                .matches(predictions -> DoubleStream.of(predictions).sum() == 1.0);

        final int classIndex = model.classify(dummyInstance);
        assertThat(classDistribution[classIndex]).as(
                "The classify method returns the index of the greatest score in the class distribution")
                .isGreaterThanOrEqualTo(classDistribution[1 - classIndex]);
    }

    /**
     * Tests the handling of a bug in H2O where a serialized Isolation Forest model trained without out-of-bag instances
     * fails its deserialization due to a {@link NumberFormatException}.
     */
    @Test
    public final void testIsolationForestWithNotEnoughInstances() {
        final H2OModelCreator loader = getMachineLearningModelLoader(H2OAlgorithm.ISOLATION_FOREST);
        final Map<String, String> params = H2OAlgorithmTestParams.getIsolationForest();
        final int sampleSize = Optional.ofNullable(params.get("sample_size")).map(Integer::parseInt).orElse(256);

        final Random random = new Random(234);
        final Dataset dataset = createDataset(SCHEMA_NO_TARGET_VARIABLE, sampleSize / 2);

        assertThatCode(() -> loader.fit(dataset, random, params))
                .as("The training of a model with no out of bag instances").doesNotThrowAnyException();
    }

    /**
     * Tests that the right exception is thrown when the dataset is empty
     *
     * @since 1.0.9
     */
    @Test
    public final void testExceptionIsThrownWhenDatasetIsEmpty() {
        final Dataset dataset = createDataset(SCHEMA, 0);
        final Map<String, String> params = H2OAlgorithmTestParams.getIsolationForest();
        final H2OModelCreator loader = getMachineLearningModelLoader(H2OAlgorithm.ISOLATION_FOREST);
        final Random random = new Random(234);

        assertThatThrownBy(() -> loader.fit(dataset, random, params))
                .isInstanceOf(ModelTrainingException.class)
                .hasMessageContaining("In order to generate the model the dataset cannot be empty")
                .hasMessageContaining("/tmp/"); //just to check that the URI is present in the message
    }

    /**
     * Ensures that all supported H2O algorithms have the mandatory parameters.
     *
     * @since 1.0.18
     */
    @Test
    public final void ensureH2OAlgorithmsHaveMandatoryParams() {

        final Map<MLAlgorithmEnum, Map<String, String>> trainAlgorithms = this.getTrainAlgorithms();

        assertThat(trainAlgorithms.size())
                .as("The list of training algorithms must not be null.")
                .isNotEqualTo(0);

        trainAlgorithms.forEach(
                (algorithm, params) -> validateParamsForTheAlgorithm(params, algorithm)
        );
    }

    /**
     * Validates the mandatory parameters that the algorithm needs.
     *
     * @param params    The default parameters.
     * @param algorithm The H2O algorithm.
     * @since 1.0.18
     */
    private void validateParamsForTheAlgorithm(final Map<String, String> params, final MLAlgorithmEnum algorithm) {

        final H2OModelCreator loader = getMachineLearningModelLoader(algorithm);

        final File tempDir = Files.createTempDir();
        tempDir.deleteOnExit();

        final List<ParamValidationError> paramValidationErrors = loader.validateForFit(
                tempDir.toPath(),
                this.dataset.getSchema(),
                params
        );

        assertThat(paramValidationErrors.size())
                .as("The list parameter validation errors must be null.")
                .isEqualTo(0);
    }

    /**
     * Tests H20 parameter validation for NAIVE_BAYES_CLASSIFIER, all parameters correct.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid_NAIVE_BAYES_CLASSIFIER() {
        final Map<String, String> params = H2OAlgorithmTestParams.getBayes();
        validateParamsForTheAlgorithm(params, H2OAlgorithm.NAIVE_BAYES_CLASSIFIER);
    }

    /**
     * Tests H20 parameter validation for NAIVE_BAYES_CLASSIFIER, one or more parameters invalid.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testExceptionIsThrownWhenParameterIsInvalid_NAIVE_BAYES_CLASSIFIER() {
        final Map<String, String> params = new HashMap<>(H2OAlgorithmTestParams.getBayes());
        params.put("laplace", "-0.25"); //must be positive

        assertExceptionIsThrownInvalidParam(params, H2OAlgorithm.NAIVE_BAYES_CLASSIFIER);
    }

    /**
     * Tests H20 parameter validation for ISOLATION_FOREST, all parameters correct.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid_ISOLATION_FOREST() {
        final Map<String, String> params = H2OAlgorithmTestParams.getIsolationForest();
        validateParamsForTheAlgorithm(params, H2OAlgorithm.ISOLATION_FOREST);
    }

    /**
     * Tests H20 parameter validation for ISOLATION_FOREST, one or more parameters invalid.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testExceptionIsThrownWhenParameterIsInvalid_ISOLATION_FOREST() {
        final Map<String, String> params = new HashMap<>(H2OAlgorithmTestParams.getIsolationForest());
        params.put("mtries", "-3"); //if negative must be -1 or -2

        assertExceptionIsThrownInvalidParam(params, H2OAlgorithm.ISOLATION_FOREST);
    }

    /**
     * Tests H20 parameter validation for XG_BOOST, all parameters correct.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid_XG_BOOST() {
        final Map<String, String> params = H2OAlgorithmTestParams.getXgboost();
        validateParamsForTheAlgorithm(params, H2OAlgorithm.XG_BOOST);
    }

    /**
     * Tests H20 parameter validation for XG_BOOST, one or more parameters invalid.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testExceptionIsThrownWhenParameterIsInvalid_XG_BOOST() {
        final Map<String, String> params = new HashMap<>(H2OAlgorithmTestParams.getXgboost());
        params.put("learn_rate", "2"); // must be between 0 and 1

        assertExceptionIsThrownInvalidParam(params, H2OAlgorithm.XG_BOOST);
    }

    /**
     * Tests H20 parameter validation for DEEP_LEARNING, all parameters correct.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid_DEEP_LEARNING() {
        final Map<String, String> params = H2OAlgorithmTestParams.getDeepLearning();
        validateParamsForTheAlgorithm(params, H2OAlgorithm.DEEP_LEARNING);
    }

    /**
     * Tests H20 parameter validation for DEEP_LEARNING, one or more parameters invalid.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testExceptionIsThrownWhenParameterIsInvalid_DEEP_LEARNING() {
        final Map<String, String> params = new HashMap<>(H2OAlgorithmTestParams.getDeepLearning());
        params.put("mini_batch_size", "0"); // must be >= 1

        assertExceptionIsThrownInvalidParam(params, H2OAlgorithm.DEEP_LEARNING);
    }

    /**
     * Tests H20 parameter validation for DISTRIBUTED_RANDOM_FOREST, all parameters correct.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid_DISTRIBUTED_RANDOM_FOREST() {
        final Map<String, String> params = H2OAlgorithmTestParams.getDrf();
        validateParamsForTheAlgorithm(params, H2OAlgorithm.DISTRIBUTED_RANDOM_FOREST);
    }

    /**
     * Tests H20 parameter validation for DISTRIBUTED_RANDOM_FOREST, one or more parameters invalid.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testExceptionIsThrownWhenParameterIsInvalid_DISTRIBUTED_RANDOM_FOREST() {
        final Map<String, String> params = new HashMap<>(H2OAlgorithmTestParams.getDrf());
        params.put("mtries", "-3"); //if negative must be -1 or -2

        assertExceptionIsThrownInvalidParam(params, H2OAlgorithm.DISTRIBUTED_RANDOM_FOREST);
    }

    /**
     * Tests H20 parameter validation for GRADIENT_BOOSTING_MACHINE, all parameters correct.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid_GRADIENT_BOOSTING_MACHINE() {
        final Map<String, String> params = H2OAlgorithmTestParams.getGbm();
        validateParamsForTheAlgorithm(params, H2OAlgorithm.GRADIENT_BOOSTING_MACHINE);
    }

    /**
     * Tests H20 parameter validation for GRADIENT_BOOSTING_MACHINE, one or more parameters invalid.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testExceptionIsThrownWhenParameterIsInvalid_GRADIENT_BOOSTING_MACHINE() {
        final Map<String, String> params = new HashMap<>(H2OAlgorithmTestParams.getGbm());
        params.put("max_abs_leafnode_pred", "0"); //must be greater than 0

        assertExceptionIsThrownInvalidParam(params, H2OAlgorithm.GRADIENT_BOOSTING_MACHINE);
    }

    /**
     * Tests H20 parameter validation for GENERALIZED_LINEAR_MODEL, all parameters correct.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid_GENERALIZED_LINEAR_MODEL() {
        final Map<String, String> params = H2OAlgorithmTestParams.getGlm();
        validateParamsForTheAlgorithm(params, H2OAlgorithm.GENERALIZED_LINEAR_MODEL);
    }

    /**
     * Tests H20 parameter validation for GENERALIZED_LINEAR_MODEL, one or more parameters invalid.
     *
     * @since @@@feedzai.next.release@@@
     */
    @Test
    public final void testExceptionIsThrownWhenParameterIsInvalid_GENERALIZED_LINEAR_MODEL() {
        final Map<String, String> params = new HashMap<>(H2OAlgorithmTestParams.getGlm());
        params.put("alpha", "2"); //must be in the range [0, 1]

        assertExceptionIsThrownInvalidParam(params, H2OAlgorithm.GENERALIZED_LINEAR_MODEL);
    }

    /**
     * Calls the validateParams method and asserts that an exception was thrown.
     *
     * @param params params to validate.
     * @param h2OAlgorithm algorithm descriptor.
     *
     * @since @@@feedzai.next.release@@@
     */
    private void assertExceptionIsThrownInvalidParam(final Map<String, String> params, final H2OAlgorithm h2OAlgorithm) {
        assertThatThrownBy(() -> validateParamsForTheAlgorithm(params, h2OAlgorithm))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Model has errors:");
    }
}
