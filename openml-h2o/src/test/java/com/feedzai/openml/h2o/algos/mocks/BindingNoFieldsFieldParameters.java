package com.feedzai.openml.h2o.algos.mocks;

import com.google.gson.annotations.SerializedName;
import water.bindings.pojos.ModelParametersSchemaV3;

/**
 * Mocked class that extends {@link ModelParametersSchemaV3} to match {@link NoFieldsFieldParameters}.
 *
 * @author Miguel Cruz (miguel.cruz@feedzai.com)
 * @since 1.0.7
 */
public class BindingNoFieldsFieldParameters extends ModelParametersSchemaV3 {
    @SerializedName("a_boolean_field")
    public boolean aBooleanField = false;

    @SerializedName("an_int_field")
    public int anIntField = 3;
}
