package com.feedzai.openml.h2o.algos.mocks;

import com.google.gson.annotations.SerializedName;
import water.bindings.pojos.ModelParametersSchemaV3;

/**
 * FIXME
 *
 * @author Miguel Cruz (miguel.cruz@feedzai.com)
 * @since @@@feedzai.next.release@@@
 */
public class BindingRegularParameters extends ModelParametersSchemaV3 {

    @SerializedName("field_1")
    public int field1;

    @SerializedName("field_2")
    public int field2;

    @SerializedName("field_3")
    public int field3;
}
