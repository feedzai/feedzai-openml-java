package com.feedzai.openml.h2o.algos.mocks;

import com.google.common.collect.ImmutableList;
import water.api.API;
import water.api.schemas3.ModelParametersSchemaV3;

import java.util.Collections;
import java.util.List;

/**
 * Mocked class that extends {@link water.bindings.pojos.ModelParametersSchemaV3} that has a {@code fields} field that
 * is not a String array.
 *
 * @author Miguel Cruz (miguel.cruz@feedzai.com)
 * @since 1.0.7
 */
public class FieldsNotArrayParameters extends ModelParametersSchemaV3 {

    public static final List<String> fields = Collections.emptyList();

    @API(
            level = API.Level.critical,
            direction = API.Direction.INPUT,
            gridable = true,
            help = "A boolean"
    )
    public boolean a_boolean_field;

    @API(
            level = API.Level.secondary,
            direction = API.Direction.INPUT,
            gridable = true,
            help = "An int"
    )
    public int an_int_field;

}
