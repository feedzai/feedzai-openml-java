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

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.data.schema.AbstractValueSchema;
import com.google.common.collect.ImmutableMap;
import water.fvec.Vec;

import java.util.List;
import java.util.Map;

/**
 * Entity responsible for converting from OpenML data representations to H2O data representations.
 *
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 * @since 0.1.0
 */
public final class H2OConverter {

    /**
     * Mapping between {@link AbstractValueSchema} implementations and H2O representations.
     */
    private static final Map<Class, String> VALUE_SCHEMA_MAPPINGS = ImmutableMap.of(
            CategoricalValueSchema.class, Vec.TYPE_STR[Vec.T_CAT],
            NumericValueSchema.class, Vec.TYPE_STR[Vec.T_NUM],
            StringValueSchema.class, Vec.TYPE_STR[Vec.T_STR]
    );

    /**
     * Empty constructor.
     */
    private H2OConverter() {
    }

    /**
     * Converts from the OpenML representation into the H2O representation of column types.
     *
     * @param fieldSchemas The {@link FieldSchema 's} holding the necessary information for converting the types.
     * @return The H2O representation of column types.
     */
    public static String[] convertColumnTypes(final List<FieldSchema> fieldSchemas) {
        return fieldSchemas
                .stream()
                .map(fieldSchema -> VALUE_SCHEMA_MAPPINGS.get(fieldSchema.getValueSchema().getClass()))
                .toArray(String[]::new);
    }

    /**
     * Converts from the OpenML representation into the H2O representation of column names.
     *
     * @param fieldSchemas The {@link FieldSchema}s holding the necessary information for converting the types.
     * @return The H2O representation of column names.
     */
    public static String[] convertColumnNames(final List<FieldSchema> fieldSchemas) {
        return fieldSchemas
                .stream()
                .map(FieldSchema::getFieldName)
                .toArray(String[]::new);
    }

    /**
     * Converts from the OpenML representation into the H2O representation of column domains.
     *
     * @param fieldSchemas The {@link FieldSchema}s holding the necessary information for converting the types.
     * @return The H2O representation of column domains.
     */
    public static String[][] convertDomains(final List<FieldSchema> fieldSchemas) {
        return fieldSchemas
                .stream()
                .map(fieldSchema -> {
                    if (fieldSchema.getValueSchema() instanceof CategoricalValueSchema) {
                        final CategoricalValueSchema valueSchema = (CategoricalValueSchema) fieldSchema.getValueSchema();
                        return valueSchema.getNominalValues().toArray(new String[]{});
                    } else {
                        return null; // H2O expects null for non-categorical/enums values: ParseSetup#_domains
                    }
                })
                .toArray(String[][]::new);
    }
}
