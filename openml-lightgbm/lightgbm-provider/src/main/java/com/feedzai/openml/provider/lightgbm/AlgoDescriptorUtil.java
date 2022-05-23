package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;

public abstract class AlgoDescriptorUtil {

    /**
     * An alias to ease the readability of parameters' configuration that are not mandatory.
     */
    protected static final boolean NOT_MANDATORY = false;

    /**
     * An alias to ease the readability of parameters' configuration that are not mandatory.
     */
    protected static final boolean MANDATORY = true;

    /**
     * Helper method to return a range of type DOUBLE.
     *
     * @param minValue Minimum allowed value.
     * @param maxValue Maximum allowed value.
     * @param defaultValue Default value.
     * @return Double range with the specs above.
     */
    protected static NumericFieldType doubleRange(final double minValue,
                                                  final double maxValue,
                                                  final double defaultValue) {
        return NumericFieldType.range(minValue, maxValue, NumericFieldType.ParameterConfigType.DOUBLE, defaultValue);
    }

    /**
     * Helper method to return a range of type INT.
     *
     * @param minValue Minimum allowed value.
     * @param maxValue Maximum allowed value.
     * @param defaultValue Default value.
     * @return Integer range with the specs above.
     */
    protected static NumericFieldType intRange(final int minValue,
                                               final int maxValue,
                                               final int defaultValue) {
        return NumericFieldType.range(minValue, maxValue, NumericFieldType.ParameterConfigType.INT, defaultValue);
    }

}
