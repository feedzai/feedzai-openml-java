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
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import java.util.List;

/**
 * Class to hold all used test schemas.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 */
public class TestSchemas {

    /**
     * Label field to be used for the numericals-only schema.
     * <p>
     * This schema fields order match the TRAIN schema in the python reference implementation,
     * where the label was as well placed as the last field.
     *
     * @see TestSchemas#NUMERICAL_SCHEMA_FEATURES
     */
    static final FieldSchema FRAUD_LABEL_INDEXED_FIELD = new FieldSchema(
            "is_fraud_label_indexed",
            4,
            new CategoricalValueSchema(true, ImmutableSet.of("0.0", "1.0"))
    );

    /**
     * Non-binary target field ( to test behaviour with non-binary targets ).
     */
    static final FieldSchema NON_BINARY_TARGET_FIELD = new FieldSchema(
            "target_field",
            0,
            new CategoricalValueSchema(true, ImmutableSet.of("A", "B", "C"))
    );

    /**
     * List of feature fields used to train the basic model with exclusively numerical features.
     * <p>
     * This schema fields order match the TRAIN schema in the python reference implementation,
     * where the label was as well placed as the last field.
     *
     * @see TestSchemas#FRAUD_LABEL_INDEXED_FIELD
     */
    private static final List<FieldSchema> NUMERICAL_SCHEMA_FEATURES = ImmutableList.<FieldSchema>builder()
            .add(new FieldSchema("amount", 0, new NumericValueSchema(false)))
            .add(new FieldSchema("num1_float", 1, new NumericValueSchema(false)))
            .add(new FieldSchema("num2_double", 2, new NumericValueSchema(false)))
            .add(new FieldSchema("num3_int", 3, new NumericValueSchema(false)))
            .build();

    /**
     * List of feature fields used to train the generic model with numerical + categorial features.
     */
    private static final List<FieldSchema> CATEGORICAL_SCHEMA_FEATURES = ImmutableList.<FieldSchema>builder()
            .addAll(NUMERICAL_SCHEMA_FEATURES)
            .add(new FieldSchema("cat1_string", 4, new CategoricalValueSchema(true,
                            ImmutableSet.of("aaa", "b", "C", "dDd"))))
            .add(new FieldSchema("cat2_string", 5, new CategoricalValueSchema(true,
                            ImmutableSet.of("aaacat2", "bcat2", "Ccat2", "dDdcat2"))))
            .add(new FieldSchema("cat3_string", 6, new CategoricalValueSchema(true,
                            ImmutableSet.of("aaacat3", "bcat3", "Ccat3", "dDdcat3", "sdofij", "blahblah"))))
            .build();

    /**
     * Model was created with numerical fields only:
     * ["num1_float", "num2_double", "num3_int", "amount"]
     * <p>
     * This schema fields order match the TRAIN schema in the python reference implementation,
     * where the label was as well placed as the last field.
     */
    public static final DatasetSchema NUMERICALS_SCHEMA_WITH_LABEL_AT_END = new DatasetSchema(
            4,
            ImmutableList.<FieldSchema>builder()
            .addAll(NUMERICAL_SCHEMA_FEATURES)
            .add(SchemaUtils.getFieldCopyWithIndex(FRAUD_LABEL_INDEXED_FIELD, 4))
            .build()
    );

    /**
     * Test schema with categoricals and label as last field in instance.
     *
     * Raw data columns @ test_data/in_train_val.csv: "card","amount","event_timestamp","is_fraud_label",
     * "uuid","cat1_generator","cat2_generator","cat3_generator","num1_float","num2_double","num3_int",
     * "cat1_string","cat2_string","cat3_string"
     */
    static final DatasetSchema CATEGORICALS_SCHEMA_LABEL_AT_END = new DatasetSchema(
            7,
            ImmutableList.<FieldSchema>builder()
                    .addAll(CATEGORICAL_SCHEMA_FEATURES)
                    .add(new FieldSchema("is_fraud_label", 7,
                            new CategoricalValueSchema(true, ImmutableSet.of("FALSE", "TRUE"))))
                    .build()
    );

    /**
     * Place the label in the middle to guarantee tests work the same.
     * Feature order should be the same.
     * @see TestSchemas#NUMERICALS_SCHEMA_WITH_LABEL_AT_END
     */
    static final DatasetSchema NUMERICALS_SCHEMA_WITH_TARGET_IN_MIDDLE = SchemaUtils.getSchemaWithTargetAt(
            NUMERICALS_SCHEMA_WITH_LABEL_AT_END, 2
    );

    /**
     * Place the label at the beginning to guarantee tests work the same.
     * Feature order should be the same.
     * @see TestSchemas#NUMERICALS_SCHEMA_WITH_LABEL_AT_END
     */
    static final DatasetSchema NUMERICALS_SCHEMA_WITH_TARGET_AT_START = SchemaUtils.getSchemaWithTargetAt(
            NUMERICALS_SCHEMA_WITH_LABEL_AT_END, 0
    );

    /**
     * A wrong test schema that lacks lots of the schema fields.
     */
    static final DatasetSchema BAD_NUMERICALS_SCHEMA_WITH_MISSING_FIELDS = new DatasetSchema(
            1,
            ImmutableList.of(
                    new FieldSchema("num2_double", 0,
                            new NumericValueSchema(false)),
                    new FieldSchema("fraud_label", 1,
                            new CategoricalValueSchema(true, ImmutableSet.of("0.0", "1.0")))
            )
    );

    /**
     * Permutate the order of features of the train schema.
     * @see TestSchemas#NUMERICALS_SCHEMA_WITH_LABEL_AT_END
     */
    static final DatasetSchema BAD_NUMERICALS_SCHEMA_WITH_WRONG_FEATURES_ORDER = new DatasetSchema(
            4,
            ImmutableList.of(
                    SchemaUtils.getFieldCopyWithIndex(NUMERICAL_SCHEMA_FEATURES.get(3), 0),
                    SchemaUtils.getFieldCopyWithIndex(NUMERICAL_SCHEMA_FEATURES.get(0), 1),
                    SchemaUtils.getFieldCopyWithIndex(NUMERICAL_SCHEMA_FEATURES.get(2), 2),
                    SchemaUtils.getFieldCopyWithIndex(NUMERICAL_SCHEMA_FEATURES.get(1), 3),
                    SchemaUtils.getFieldCopyWithIndex(FRAUD_LABEL_INDEXED_FIELD, 4)
            )
    );

    /**
     * Derived test schema with categoricals and label between the feature fields.
     */
    static final DatasetSchema CATEGORICALS_SCHEMA_LABEL_IN_MIDDLE = SchemaUtils.getSchemaWithTargetAt(
            CATEGORICALS_SCHEMA_LABEL_AT_END, 1
    );

    /**
     * Derived test schema with categoricals and label as first field in instance.
     */
    static final DatasetSchema CATEGORICALS_SCHEMA_LABEL_AT_START = SchemaUtils.getSchemaWithTargetAt(
            CATEGORICALS_SCHEMA_LABEL_AT_END, 0
    );

    /**
     * Schema with non-binary classification target.
     */
    static final DatasetSchema SCHEMA_WITH_NON_BINARY_CLASSIFICATION_TARGET = new DatasetSchema(
            1,
            ImmutableList.of(
                    NUMERICAL_SCHEMA_FEATURES.get(0),
                    SchemaUtils.getFieldCopyWithIndex(NON_BINARY_TARGET_FIELD, 1)
            )
    );

    /**
     * Schema with no target.
     */
    static final DatasetSchema SCHEMA_WITH_NO_TARGET = new DatasetSchema(
            -1,
            TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END.getPredictiveFields()
    );

    /**
     * Schema with regression target.
     */
    static final DatasetSchema SCHEMA_WITH_REGRESSION_TARGET = new DatasetSchema(
            0,
            TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END.getPredictiveFields()
    );
}
