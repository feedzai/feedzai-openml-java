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

import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.google.common.primitives.Floats;
import hex.VarImp;
import org.assertj.core.api.SoftAssertions;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

/**
 * Tests the behaviour of {@link H2OFeatureImportanceCreator}.
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 0.1.0
 */
public class H2OFeatureImportanceCreatorTest implements H2ODatasetMixin {

    /**
     * Random class used for testing.
     */
    private Random random;

    /**
     * A common {@link H2OFeatureImportanceCreator} used for testing.
     */
    private H2OFeatureImportanceCreator featureImportanceCreator;

    @Before
    public void setup() {
        this.random = new Random(23);
        this.featureImportanceCreator = new H2OFeatureImportanceCreator();
    }

    /**
     * Simple test for calculating feature importance. Simply tests that the schema of the result matches the dataset schema.
     *
     * @throws ModelTrainingException if the feature importance calculation fails due to an error in model training.
     * @since 0.1.0
     */
    @Test
    public final void calculateFeatureImportance() throws ModelTrainingException {
        final MockDataset dataset = TRAIN_DATASET;

        final VarImp featureImportance = this.featureImportanceCreator.calculateFeatureImportance(dataset, this.random, H2OAlgorithmTestParams.getGbm());

        final SoftAssertions assertions = new SoftAssertions();
        assertions.assertThat(featureImportance)
                .as("The feature importance result is not null")
                .isNotNull();

        for (final FieldSchema fieldSchema : dataset.getSchema().getFieldSchemas()) {
            assertions.assertThat(featureImportance._names)
                    .as("Field %s is mentioned in the feature importance result.")
                    .contains(fieldSchema.getFieldName());
        }

        assertions.assertThat(Floats.asList(featureImportance._varimp))
                .as("The feature importance values")
                .allMatch(importance -> importance >= 0);

        assertions.assertAll();
    }

}
