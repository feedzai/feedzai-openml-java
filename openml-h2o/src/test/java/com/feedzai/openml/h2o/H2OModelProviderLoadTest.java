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

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.feedzai.openml.util.provider.AbstractProviderCategoricalTargetTest;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.assertj.core.api.Assertions;
import org.assertj.core.util.Lists;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests for loading models with {@link H2OModelProvider}.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public class H2OModelProviderLoadTest extends AbstractProviderCategoricalTargetTest<AbstractClassificationH2OModel, H2OModelCreator, H2OModelProvider> {

    /**
     * Name of the directory that contains a dummy model imported in MOJO format. The model was created with a schema
     * similar to {@link #createDatasetSchema(Set)} ()};
     */
    private static final String MOJO_MODEL_FILE = "deeplearning";

    /**
     * Name of the directory that contains a dummy model imported in POJO format. The model was created with a schema
     * similar to {@link #createDatasetSchema(Set)} ()};
     */
    public static final String POJO_MODEL_FILE = "drf";

    /**
     * The values of the target variable.
     */
    private static final Set<String> TARGET_VALUES = ImmutableSet.of("true", "false");

    @Override
    public Instance getDummyInstance() {
        return new MockInstance(new double[]{3423432.0, 7.0, 0.0});
    }

    @Override
    public Instance getDummyInstanceDifferentResult() {
        return new MockInstance(new double[]{3223434.0, 6.0, 0.0});
    }

    @Override
    public AbstractClassificationH2OModel getFirstModel() throws ModelLoadingException {
        return loadModel(H2OAlgorithm.DEEP_LEARNING, MOJO_MODEL_FILE, TARGET_VALUES);
    }

    @Override
    public AbstractClassificationH2OModel getSecondModel() throws ModelLoadingException {
        return loadModel(H2OAlgorithm.DISTRIBUTED_RANDOM_FOREST, POJO_MODEL_FILE, TARGET_VALUES);
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
    public H2OModelCreator getFirstMachineLearningModelLoader() {
        return getMachineLearningModelLoader(H2OAlgorithm.DEEP_LEARNING);
    }

    @Override
    public H2OModelProvider getMachineLearningProvider() {
        return new H2OModelProvider();
    }

    @Override
    public MLAlgorithmEnum getValidAlgorithm() {
        return H2OAlgorithm.DEEP_LEARNING;
    }

    @Override
    public String getValidModelDirName() {
        return MOJO_MODEL_FILE;
    }

    @Override
    public Set<String> getFirstModelTargetNominalValues() {
        return TARGET_VALUES;
    }

    @Override
    public DatasetSchema createDatasetSchema(final Set<String> targetValues) {
        return new DatasetSchema(
                2,
                ImmutableList.<FieldSchema>builder()
                        .add(
                                new FieldSchema(
                                        "date",
                                        0,
                                        new NumericValueSchema(false)
                                )
                        )
                        .add(
                                new FieldSchema(
                                        "amount",
                                        1,
                                        new NumericValueSchema(false)
                                )
                        )
                        .add(
                                new FieldSchema(
                                        "fraud",
                                        2,
                                        new CategoricalValueSchema(false, targetValues)
                                )
                        )
                        .build()
        );
    }

    /**
     * Tests that when loading a model with a given field set A that is linked to a given {@link DatasetSchema} B, where both schemas are not compatible,
     * the loading fails with a {@link ModelLoadingException}.
     */
    @Test
    public final void testLoadModelWithPartialSchema() {
        final DatasetSchema fullSchema = createDatasetSchema(TARGET_VALUES);
        final DatasetSchema partialSchema = removeNonTargetVariable(fullSchema);

        final H2OModelCreator modelLoader = getMachineLearningModelLoader(H2OAlgorithm.DISTRIBUTED_RANDOM_FOREST);
        final String modelPath = this.getClass().getResource("/" + POJO_MODEL_FILE).getPath();

        Assertions.assertThatThrownBy(() -> modelLoader.loadModel(Paths.get(modelPath), partialSchema))
                .isInstanceOf(ModelLoadingException.class);

    }

    /**
     * Returns a {@link DatasetSchema} copied from the schema provided, where one of the fields is removed. The removed field is never the one
     * assigned as target variable.
     *
     * @param fullSchema The schema from which the new copy is generated.
     * @return The schema with one less field.
     */
    private DatasetSchema removeNonTargetVariable(final DatasetSchema fullSchema) {
        final List<FieldSchema> fullFields = Lists.newArrayList(fullSchema.getFieldSchemas());
        if (fullFields.size() < 2) {
            throw new IllegalStateException("This schema does not have enough fields for this test to work.");
        }
        final int lastIndex = fullFields.size() - 1;

        final int indexToRemove = fullSchema.getTargetIndex()
                .map(targetIndex -> targetIndex == lastIndex ? lastIndex - 1 : lastIndex)
                .orElse(0);

        int index = 0;
        final List<FieldSchema> fields = new ArrayList<>(fullFields.size() - 1);
        for (int i = 0; i < fullFields.size(); i++) {
            if (i != indexToRemove) {
                final FieldSchema field = fullFields.get(i);
                fields.add(new FieldSchema(field.getFieldName(), index++, field.getValueSchema()));
            }
        }

        return fullSchema.getTargetIndex()
                // calculate the new target index
                .map(targetIndex -> targetIndex == lastIndex ? lastIndex - 1 : lastIndex)
                .map(targetIndex -> new DatasetSchema(targetIndex, fields))
                .orElseGet(() -> new DatasetSchema(fields));
    }
}
