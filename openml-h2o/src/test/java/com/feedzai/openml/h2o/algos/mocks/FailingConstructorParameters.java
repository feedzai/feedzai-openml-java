package com.feedzai.openml.h2o.algos.mocks;

import com.google.gson.annotations.SerializedName;
import water.bindings.pojos.ModelParametersSchemaV3;

/**
 * Mocked class that extends {@link ModelParametersSchemaV3} with matching fields
 * but a constructor that always throws, used to test the exception path in
 * {@code ParametersBuilderUtil.getParamsInstance()}.
 *
 * @since 2.0.2
 */
public class FailingConstructorParameters extends ModelParametersSchemaV3 {

    @SerializedName("field_1")
    public int field1;

    @SerializedName("field_2")
    public int field2;

    public FailingConstructorParameters() {
        throw new RuntimeException("Simulated constructor failure");
    }
}
