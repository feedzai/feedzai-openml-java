package com.feedzai.openml.h2o.algos.mocks;

import water.api.API;
import water.api.schemas3.ModelParametersSchemaV3;

/**
 * FIXME
 *
 * @author Miguel Cruz (miguel.cruz@feedzai.com)
 * @since @@@feedzai.next.release@@@
 */
public class RegularParameters extends ModelParametersSchemaV3 {

    /**
     * Private field used for tests.
     */
    public static final String[] fields = new String[]{"field_1", "field_2"};

    @API(
            level = API.Level.critical,
            direction = API.Direction.INPUT,
            gridable = true,
            help = "field 1"
    )
    public int field_1;

    @API(
            level = API.Level.secondary,
            direction = API.Direction.INPUT,
            gridable = true,
            help = "field 2"
    )
    public int field_2;

    @API(
            level = API.Level.secondary,
            direction = API.Direction.INPUT,
            gridable = true,
            help = "field 3"
    )
    public int field_3;
}
