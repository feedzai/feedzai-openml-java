package com.feedzai.openml.provider.lightgbm.resources.schemas;

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import java.util.List;

/**
 * Schemas with soft labels.
 *
 * @author alberto.ferreira
 */
public class SoftSchemas {

    public static final List<FieldSchema> SOFT_SCHEMA_FIELDS = ImmutableList.<FieldSchema>builder()
            .add(new FieldSchema("f_float", 0, new NumericValueSchema(false)))
            .add(new FieldSchema("f_cat", 1, new CategoricalValueSchema(false, ImmutableSet.of("a", "b"))))
            .add(new FieldSchema("f_int", 2, new NumericValueSchema(false)))
            .add(new FieldSchema("soft", 3, new NumericValueSchema(false)))
            .add(new FieldSchema("soft_uninformative", 4, new NumericValueSchema(false)))
            .add(new FieldSchema("hard", 5, new CategoricalValueSchema(true, ImmutableSet.of("0", "1"))))
            .add(new FieldSchema("tempo_ms", 6, new NumericValueSchema(false)))
            .build();

    public static final DatasetSchema SOFT_SCHEMA = new DatasetSchema(
            5,
            ImmutableList.<FieldSchema>builder()
                    .addAll(SOFT_SCHEMA_FIELDS)
                    .build()
    );

}
