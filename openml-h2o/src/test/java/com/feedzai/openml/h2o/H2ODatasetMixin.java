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

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.google.common.collect.ImmutableSet;

import java.util.Random;

/**
 * Mixin interface with common utilities for H2O dataset-related operations.
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 0.1.0
 */
public interface H2ODatasetMixin {

    /**
     * Possible values for the Categorical target variable of the test schema.
     */
    ImmutableSet<String> TARGET_VALUES = ImmutableSet.of("a", "b", "c");

    /**
     * Schema to use in the tests.
     */
    DatasetSchema SCHEMA = MockDataset.generateDefaultSchema(TARGET_VALUES, 4);

    /**
     * Example schema that does not use target variable.
     */
    DatasetSchema SCHEMA_NO_TARGET_VARIABLE = new DatasetSchema(SCHEMA.getFieldSchemas());

    /**
     * Number of instances to have in the train dataset.
     */
    int TRAIN_DATASET_SIZE = 50;

    /**
     * Dataset to use to train models.
     */
    MockDataset TRAIN_DATASET = new MockDataset(SCHEMA, TRAIN_DATASET_SIZE, new Random(0));

}
