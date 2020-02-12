package com.feedzai.openml.h2o.algos.mocks;

import com.google.gson.annotations.SerializedName;
import water.bindings.pojos.ModelParametersSchemaV3;

/**
 * Mocked class that extends {@link ModelParametersSchemaV3} to match {@link PrivateFieldsFieldParameters}.
 *
 * @author Miguel Cruz (miguel.cruz@feedzai.com)
 * @since 1.0.7
 */
public class BindingPrivateFieldsFieldParameters extends ModelParametersSchemaV3 {

    @SerializedName("field_1")
    public int field1;

    @SerializedName("field_2")
    public int field2;
}
