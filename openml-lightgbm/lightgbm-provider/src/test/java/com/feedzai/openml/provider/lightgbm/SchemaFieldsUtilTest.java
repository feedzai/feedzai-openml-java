/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * © 2026 Feedzai, Strictly Confidential
 */

package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for {@link SchemaFieldsUtil}.
 *
 * @author Joaquim Leitão (joaquim.leitao@feedzai.com)
 */
public class SchemaFieldsUtilTest {

    /**
     * Schema: amount(0), num1(1), label(2), num2(3)
     * Target index: 2
     */
    private static final DatasetSchema SCHEMA_LABEL_IN_MIDDLE = new DatasetSchema(
            2,
            ImmutableList.of(
                    new FieldSchema("amount", 0, new NumericValueSchema(false)),
                    new FieldSchema("num1", 1, new NumericValueSchema(false)),
                    new FieldSchema("label", 2, new CategoricalValueSchema(true, ImmutableSet.of("0", "1"))),
                    new FieldSchema("num2", 3, new NumericValueSchema(false))
            )
    );

    /**
     * Tests that {@link SchemaFieldsUtil#getColumnIndex} correctly identifies the index
     * of a field that is present in the provided schema.
     */
    @Test
    public void testGetColumnIndexExistingField() {
        final Optional<Integer> result = SchemaFieldsUtil.getColumnIndex("amount", SCHEMA_LABEL_IN_MIDDLE);
        assertThat(result).isPresent().contains(0);
    }

    /**
     * Tests that {@link SchemaFieldsUtil#getColumnIndex} correctly identifies the index
     * of a field that is present in the provided schema, in a case-insensitive manner.
     */
    @Test
    public void testGetColumnIndexCaseInsensitive() {
        final Optional<Integer> result = SchemaFieldsUtil.getColumnIndex("AMOUNT", SCHEMA_LABEL_IN_MIDDLE);
        assertThat(result).isPresent().contains(0);
    }

    /**
     * Tests that {@link SchemaFieldsUtil#getColumnIndex} returns an empty {@link Optional}
     * when the requested field name is not present in the schema.
     */
    @Test
    public void testGetColumnIndexNotFound() {
        final Optional<Integer> result = SchemaFieldsUtil.getColumnIndex("nonexistent_field", SCHEMA_LABEL_IN_MIDDLE);
        assertThat(result).isEmpty();
    }

    /**
     * Test that when a field comes before the target column in the schema,
     * {@link SchemaFieldsUtil#getColumnIndex} returns its index unchanged
     * (LightGBM's internal representation excludes the target column, so
     * fields before the target should retain their index unchanged).
     */
    @Test
    public void testGetFieldIndexWithoutLabelFieldBeforeTarget() {
        final Optional<Integer> result = SchemaFieldsUtil.getFieldIndexWithoutLabel("num1", SCHEMA_LABEL_IN_MIDDLE);
        assertThat(result).isPresent().contains(1);
    }

    /**
     * Test that when a field comes after the target column in the schema,
     * {@link SchemaFieldsUtil#getColumnIndex} offsets the field index by -1
     * (LightGBM's internal representation excludes the target column, so
     * fields after the target have their index offset by -1).
     */
    @Test
    public void testGetFieldIndexWithoutLabelFieldAfterTarget() {
        final Optional<Integer> result = SchemaFieldsUtil.getFieldIndexWithoutLabel("num2", SCHEMA_LABEL_IN_MIDDLE);
        assertThat(result).isPresent().contains(2);
    }

    /**
     * Tests that {@link SchemaFieldsUtil#getFieldIndexWithoutLabel} returns
     * an empty {@link Optional} when the requested field name is not present
     * in the schema.
     */
    @Test
    public void testGetFieldIndexWithoutLabelNotFound() {
        final Optional<Integer> result = SchemaFieldsUtil.getFieldIndexWithoutLabel("nonexistent", SCHEMA_LABEL_IN_MIDDLE);
        assertThat(result).isEmpty();
    }

    /**
     * Tests that {@link SchemaFieldsUtil#getFieldIndexWithoutLabel} throws
     * a RuntimeException when no target is specified in the schema.
     */
    @Test(expected = RuntimeException.class)
    public void testGetFieldIndexWithoutLabelNoTarget() {
        // Schema without a target index
        final DatasetSchema schemaNoTarget = new DatasetSchema(
                ImmutableList.of(
                        new FieldSchema("amount", 0, new NumericValueSchema(false)),
                        new FieldSchema("num1", 1, new NumericValueSchema(false))
                )
        );
        SchemaFieldsUtil.getFieldIndexWithoutLabel("amount", schemaNoTarget);
    }
}
