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
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;

/**
 * Tests for training models with {@link H2OModelProvider}.
 *
 * @author Luis Reis (luis.reis@feedzai.com)
 * @since 0.1.0
 */
public class H2OModelProviderTrainTest extends AbstractProviderModelTrainTest<ClassificationH2OModel, H2OModelCreator, H2OModelProvider> implements H2ODatasetMixin {

    /**
     * Logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OModelProviderTrainTest.class);

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
    public ClassificationH2OModel getFirstModel() throws ModelTrainingException {
        final H2OModelCreator modelCreator = getMachineLearningModelLoader(H2OAlgorithm.DEEP_LEARNING);
        return modelCreator.fit(TRAIN_DATASET, new Random(0), ImmutableMap.of());
    }

    @Override
    public ClassificationH2OModel getSecondModel() throws ModelTrainingException {
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
        return TRAIN_DATASET;
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

        return builder.build();
    }


    /**
     * Tests that the right exception is thrown when the dataset is empty
     *
     * @since 1.0.9
     */
    @Test
    public final void testExceptionIsThrownWhenDatasetIsEmpty() {
        final Dataset dataset = createDataset(SCHEMA, 0);
        final Map<String, String> params = H2OAlgorithmTestParams.getDeepLearning();
        final H2OModelCreator loader = getMachineLearningModelLoader(H2OAlgorithm.DEEP_LEARNING);
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
                TRAIN_DATASET.getSchema(),
                params
        );

        assertThat(paramValidationErrors.size())
                .as("The list parameter validation errors must be null.")
                .isEqualTo(0);
    }
}
