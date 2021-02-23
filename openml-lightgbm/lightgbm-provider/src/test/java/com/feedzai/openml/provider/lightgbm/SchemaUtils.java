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

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

/**
 * Class with utils for schema manipulation.
 * <p>
 * Adheres to OpenML Dataset and DataSchema API.
 * <p>
 * Parameter conventions of such API's are mantained,
 * e.g., targetFieldIndex=-1 => no target field.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
class SchemaUtils {

    /**
     * Returns the field but with the field index replaced.
     *
     * @param field      input field
     * @param fieldIndex index to change the field's fieldIndex to
     * @return new field with the indexField matching fieldIndex.
     */
    static FieldSchema getFieldCopyWithIndex(final FieldSchema field, final int fieldIndex) {

        return new FieldSchema(
                field.getFieldName(),
                fieldIndex,
                field.getValueSchema()
        );
    }

    /**
     * Returns a new schema but with the target field index moved/(removed) to the target position.
     *
     * @param schema            Input schema (with or without label)
     * @param outputTargetIndex Final index of the target field.
     *                          <p>
     *                          Several options are available:
     *                          <p>
     *                          n => place at that exact position
     *                          n (where n > #fields) => automatically placed as last field in index = #fields-1
     *                          null => place the label at the end just like the option above
     *                          -1 => output schema without label, following the OpenML Dataset creation standard.
     * @return schema with the label at the desired position (or no label if requested).
     * @warning outputTargetIndex >= 0 will throw a RuntimeException if the input schema has no label.
     */
    static DatasetSchema getSchemaWithTargetAt(final DatasetSchema schema, final int outputTargetIndex) {

        final int originalTargetIndex = schema.getTargetIndex().orElse(-1);
        final List<FieldSchema> fields = schema.getFieldSchemas();
        final int numFields = fields.size();

        // Fix the user choice to match the proper end index if necessary:
        final int outputTargetFieldIndex = (outputTargetIndex < numFields) ? outputTargetIndex : numFields - 1;

        if (outputTargetFieldIndex < 0) {
            return getSchemaWithoutLabel(schema);
        } else if (originalTargetIndex == -1) {
            throw new RuntimeException("Input schema has no label. Invalid operation.");
        } else {
            // Both input and output schemas have labelindex >= 0.

            final int labelShiftRegionMin = Math.min(originalTargetIndex, outputTargetFieldIndex);
            final int labelShiftRegionMax = Math.max(originalTargetIndex, outputTargetFieldIndex);

            final List<FieldSchema> outputFields = new ArrayList<>();
            for (int i = 0; i < numFields; ++i) {

                int fetchOffset = 0;
                if (labelShiftRegionMin <= i && i <= labelShiftRegionMax) {
                    if (i == outputTargetFieldIndex) {
                        fetchOffset = outputTargetFieldIndex - originalTargetIndex;
                    } else {
                        fetchOffset = (i < outputTargetFieldIndex) ? -1 : 1;
                    }
                }

                outputFields.add(getFieldCopyWithIndex(fields.get(i - fetchOffset), i));
            }

            return new DatasetSchema(outputTargetFieldIndex, outputFields);
        }
    }

    /**
     * Generates a schema that has no target field.
     *
     * @param inputSchema Input schema with or without target field.
     * @return Schema without target field.
     */
    static DatasetSchema getSchemaWithoutLabel(final DatasetSchema inputSchema) {

        final List<FieldSchema> features = inputSchema.getPredictiveFields();

        return new DatasetSchema(
                IntStream.range(0, features.size())
                        .mapToObj(i -> getFieldCopyWithIndex(features.get(i), i))
                        .collect(toList())
        );
    }
}
