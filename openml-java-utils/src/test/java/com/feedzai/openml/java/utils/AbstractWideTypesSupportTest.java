/*
 * Copyright 2019 Feedzai
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
package com.feedzai.openml.java.utils;

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.model.MachineLearningModelLoader;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Generic regression test for PULSEDEV-27879 where we confirm that Java providers support all type fields.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 1.0.5
 */
public abstract class AbstractWideTypesSupportTest {

    /**
     * Confirms that a model is valid even if its dataset schema has a String feature field.
     */
    @Test
    public void testAllTypes() {
        final DatasetSchema schema = new DatasetSchema(1, ImmutableList.of(
                new FieldSchema("numFeature", 0, new NumericValueSchema(false)),
                new FieldSchema("catFeature", 1, new CategoricalValueSchema(true, ImmutableSet.of("A", "B"))),
                new FieldSchema("strFeature", 2, new StringValueSchema(true))
        ));

        final List<ParamValidationError> errors =
                getModelLoader().validateForLoad(Paths.get(getModelPath()), schema, ImmutableMap.of());

        assertThat(errors).isEmpty();
    }

    /**
     * Gets the path to the model to use.
     *
     * @return The path.
     */
    protected abstract String getModelPath();

    /**
     * Gets the model loader to use.
     *
     * @return The model loader.
     */
    protected abstract MachineLearningModelLoader<?> getModelLoader();

}
