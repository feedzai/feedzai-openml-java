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
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.feedzai.openml.util.provider.AbstractProviderModelTrainTest;
import com.google.common.collect.ImmutableMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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


}
