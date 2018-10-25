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

package com.feedzai.openml.datarobot;

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.feedzai.openml.util.provider.AbstractProviderCategoricalTargetTest;
import com.google.common.collect.ImmutableSet;

import java.nio.file.Paths;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Abstract class to test the loading of models with {@link DataRobotModelProvider}.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public abstract class AbstractDataRobotModelProviderLoadTest extends
        AbstractProviderCategoricalTargetTest<ClassificationBinaryDataRobotModel, DataRobotModelCreator, DataRobotModelProvider> {

    /**
     * Name of the directory that contains a dummy model trained with DataRobot (RandomForest Classifier). The model was
     * created with a schema similar to {@link #createDatasetSchema(Set)} ()};
     */
    protected static final String FIRST_MODEL_FILE = "valid_classifier";

    /**
     * Similar to {@link #FIRST_MODEL_FILE} but with a different dummy model, DataRobot (eXtreme Gradient Boosted Trees
     * Classifier).
     */
    protected static final String SECOND_MODEL_FILE = "second_valid_classifier";

    /**
     * The values of the target variable.
     */
    private static final Set<String> TARGET_VALUES = ImmutableSet.of("0", "1");

    @Override
    public Instance getDummyInstance() {
        return new MockInstance(createDatasetSchema(getFirstModelTargetNominalValues()), new Random(123));
    }

    @Override
    public Instance getDummyInstanceDifferentResult() {
        return new MockInstance(createDatasetSchema(getFirstModelTargetNominalValues()), new Random(321));
    }

    @Override
    public ClassificationBinaryDataRobotModel getFirstModel() throws ModelLoadingException {
        return loadModel(getValidAlgorithm(), getValidModelDirName(), TARGET_VALUES);
    }

    @Override
    public ClassificationBinaryDataRobotModel getSecondModel() throws ModelLoadingException {
        return loadModel(getValidAlgorithm(), SECOND_MODEL_FILE, TARGET_VALUES);
    }

    @Override
    public Set<Integer> getClassifyValuesOfFirstModel() {
        return IntStream.range(0, TARGET_VALUES.size()).boxed().collect(Collectors.toSet());
    }

    @Override
    public Set<Integer> getClassifyValuesOfSecondModel() {
        return getClassifyValuesOfFirstModel();
    }

    @Override
    public DataRobotModelCreator getFirstMachineLearningModelLoader() {
        return getMachineLearningModelLoader(getValidAlgorithm());
    }

    @Override
    public DataRobotModelProvider getMachineLearningProvider() {
        return new DataRobotModelProvider();
    }

    @Override
    public MLAlgorithmEnum getValidAlgorithm() {
        return DataRobotAlgorithm.GENERIC_BINARY_CLASSIFICATION;
    }

    @Override
    public String getValidModelDirName() {
        return FIRST_MODEL_FILE;
    }

    @Override
    public Set<String> getFirstModelTargetNominalValues() {
        return TARGET_VALUES;
    }

    @Override
    public DatasetSchema createDatasetSchema(final Set<String> targetValues) {
        final String modelPath = getClass().getResource("/" + FIRST_MODEL_FILE).getPath();
        try {
            return getFirstMachineLearningModelLoader().loadSchema(Paths.get(modelPath));
        } catch (final ModelLoadingException e) {
            throw new RuntimeException("Error loading the schema file of " + modelPath, e);
        }
    }
}
