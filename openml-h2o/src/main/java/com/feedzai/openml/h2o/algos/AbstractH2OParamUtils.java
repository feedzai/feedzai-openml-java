/*
 * Copyright 2019 Feedzai
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package com.feedzai.openml.h2o.algos;

import com.feedzai.openml.data.schema.DatasetSchema;
import org.apache.commons.lang3.StringUtils;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;

import java.util.Map;
import java.util.Optional;

/**
 * Abstract class to parse H2O algorithm params.
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
    protected abstract T parseSpecificParams(final T h2oParams, final Map<String, String> params, final long randomSeed);

    /**
     * Returns an empty representation of the algorithm specific parameters.
     *
     * @return An empty representation of the algorithm specific parameters.
     */
    protected abstract T getEmptyParams();


    /**
     * Template method to parse H2O algorithm params.
     *
     * @param trainingFrame The dataset to be used.
     * @param params        The raw training params.
     * @param randomSeed    The source of randomness.
     * @param datasetSchema The dataset schema.
     * @return The modified version of the given {@code h2oParams}.
     */
    public final T parseParams(final Frame trainingFrame,
                               final Map<String, String> params,
                               final long randomSeed,
                               final DatasetSchema datasetSchema) {
        final T baseParams = commonParams(trainingFrame, datasetSchema);
        return parseSpecificParams(baseParams, params, randomSeed);
    }

    /**
     * Auxiliary method to setup the common training params to all algorithms/models.
     *
     * @param trainingFrame The dataset to be used.
     * @param datasetSchema The index of the target variable.
     * @return A modified version of the provided params object.
     */
    protected abstract T  commonParams(Frame trainingFrame, DatasetSchema datasetSchema);

}
