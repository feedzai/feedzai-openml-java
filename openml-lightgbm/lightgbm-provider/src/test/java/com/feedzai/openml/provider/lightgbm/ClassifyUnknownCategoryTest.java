/*
 * Copyright 2020 Feedzai
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

package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.assertj.core.api.Assertions;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Based on regression test for PULSEDEV-24998 from H2O.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
public class ClassifyUnknownCategoryTest {

    /**
     * Tests that LightGBM can score an instance containing a categorical value that was not present in the training data.
     *
     * @throws ModelTrainingException If any unexpected error occurs.
     */
    @Test
    public void test() throws ModelTrainingException {

        final DatasetSchema schema = new DatasetSchema(0, ImmutableList.of(
                new FieldSchema("isFraud", 0, new CategoricalValueSchema(true, ImmutableSet.of("0", "1"))),
                new FieldSchema("catFeature", 1, new CategoricalValueSchema(true, ImmutableSet.of("A", "B")))
        ));
        final MockDataset dataset = new MockDataset(schema, ImmutableList.of(
                new MockInstance(ImmutableList.of(0.0, 0.0)),
                new MockInstance(ImmutableList.of(1.0, 0.0))
        ));
        final LightGBMBinaryClassificationModel model = new LightGBMModelCreator()
                .fit(dataset, new Random(0), TestParameters.getDefaultLightGBMParameters());

        final int predictedClass = model.classify(
                new MockInstance(ImmutableList.of(1.0, 1.0))
        );
        assertThat(predictedClass)
                .as("the predictedClass")
                .isBetween(0, 1);

        final double[] scoresDist = model.getClassDistribution(
                new MockInstance(ImmutableList.of(0.0, 1.0))
        );
        assertThat(Arrays.stream(scoresDist).sum())
                .as("sum of the class distribution", new Object[0])
                .isCloseTo(1.0D, Assertions.within(0.01D));
    }
    
}
