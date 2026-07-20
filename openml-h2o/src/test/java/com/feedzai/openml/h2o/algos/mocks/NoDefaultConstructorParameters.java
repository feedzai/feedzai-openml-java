package com.feedzai.openml.h2o.algos.mocks;

import water.bindings.pojos.ModelParametersSchemaV3;

/**
 * Mocked class that extends {@link ModelParametersSchemaV3} but has no no-arg constructor,
 * used to test the error path in {@code ParametersBuilderUtil.getParamsInstance()}.
 *
 * @since 2.0.2
 */
public class NoDefaultConstructorParameters extends ModelParametersSchemaV3 {

    public NoDefaultConstructorParameters(final String required) {
        // intentionally no default constructor
    }
}
