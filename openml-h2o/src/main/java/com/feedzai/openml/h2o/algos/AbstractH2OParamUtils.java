/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * © 2019 Feedzai, Strictly Confidential
 */

package com.feedzai.openml.h2o.algos;

import org.apache.commons.lang3.StringUtils;
import water.api.schemas3.ModelParametersSchemaV3;

import java.util.Map;
import java.util.Optional;

/**
 * Abstract class to parse H2O supervised algorithm params.
 *
 * @param <T> The concrete type of {@link ModelParametersSchemaV3 algorithm params}.
 * @since 1.0.0
 * @author Joao Sousa (joao.sousa@feedzai.com)
 */
public abstract class AbstractH2OParamUtils<T extends ModelParametersSchemaV3> {

    /**
     * Cleans a parameter value.
     *
     * @param param The raw parameter value.
     * @return An optional containing the processed param value, or an empty optional if the raw value is not meaningful.
     */
    Optional<String> cleanParam(final String param) {
        if (StringUtils.isBlank(param)) {
            return Optional.empty();
        } else {
            return Optional.of(StringUtils.trim(param));
        }
    }

    /**
     * Parses algorithm specific parameters from the raw params.
     *
     * @param h2oParams The {@link ModelParametersSchemaV3} to be filled.
     * @param params The raw training params.
     * @param randomSeed The source of randomness.
     * @return The modified version of the given {@code h2oParams}.
     */
    abstract T parseSpecificParams(final T h2oParams, final Map<String, String> params, final long randomSeed);

    /**
     * Returns an empty representation of the algorithm specific parameters.
     *
     * @return An empty representation of the algorithm specific parameters.
     */
    abstract T getEmptyParams();
}
