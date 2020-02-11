package com.feedzai.openml.h2o.algos.mocks;

import hex.deeplearning.DeepLearningModel;
import water.api.API;
import water.api.schemas3.ModelParametersSchemaV3;

/**
 * Mocked class that extends {@link water.bindings.pojos.ModelParametersSchemaV3} without having a static field named
 * {@code fields}.
 *
 * @author Miguel Cruz (miguel.cruz@feedzai.com)
 * @since 1.0.7
 */
public class NoFieldsFieldParameters extends ModelParametersSchemaV3 {

    /**
     * The serial UID.
     */
    private static final long serialVersionUID = -6778629027417552326L;

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
