/*
 * Copyright 2018 Feedzai
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

import water.api.schemas3.FrameV3;
import water.api.schemas3.KeyV3;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;

import java.util.Map;

/**
 * Abstract class to parse H2O supervised algorithm params.
 *
 * @param <T> The concrete type of {@link ModelParametersSchemaV3 algorithm params}.
 * @since 0.1.0
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 */
public abstract class AbstractSupervisedH2OParamUtils<T extends ModelParametersSchemaV3> extends AbstractH2OParamUtils<T> {

    /**
     * Auxiliary method to setup the common training params to all algorithms/models.
     *
     * @param trainingFrame The dataset to be used.
     * @param targetIndex   The index of the target variable.
     * @return A modified version of the provided params object.
     */
    private T commonParams(final Frame trainingFrame, final int targetIndex) {
        final T baseParams = getEmptyParams();
        baseParams.training_frame = new KeyV3.FrameKeyV3(trainingFrame._key);
        baseParams.ignore_const_cols = false;

        final FrameV3.ColSpecifierV3 targetVar = new FrameV3.ColSpecifierV3();
        targetVar.column_name = trainingFrame.name(targetIndex);
        baseParams.response_column = targetVar;

        return baseParams;
    }

    /**
     * Template method to parse H2O algorithm params.
     *
     * @param trainingFrame The dataset to be used.
     * @param targetIndex   The index of the target variable.
     * @param params        The raw training params.
     * @param randomSeed    The source of randomness.
     * @return The modified version of the given {@code h2oParams}.
     */
    public final T parseParams(final Frame trainingFrame,
                               final int targetIndex,
                               final Map<String, String> params,
                               final long randomSeed) {
        final T baseParams = commonParams(trainingFrame, targetIndex);
        return parseSpecificParams(baseParams, params, randomSeed);
    }

}
