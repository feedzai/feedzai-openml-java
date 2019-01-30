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

import com.feedzai.openml.h2o.params.ParametersBuilderUtil;
import com.feedzai.openml.h2o.params.ParamsValueSetter;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import hex.schemas.GBMV3.GBMParametersV3;

import java.util.Map;
import java.util.Set;

/**
 * Utility class to hold relevant information to train H2O GBM models.
 *
 * @since 0.1.0
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 */
public final class H2OGbmUtils extends AbstractSupervisedH2OParamUtils<GBMParametersV3> {

    /**
     * The set of parameters that are possible to define during the creation of an H2O GBM model.
     */
    public static final Set<ModelParameter> PARAMETERS =
            ParametersBuilderUtil.getParametersFor(GBMParametersV3.class, water.bindings.pojos.GBMParametersV3.class);

    /**
     * The setter capable of assigning a value of a parameter to the right H2O REST POJO field.
     */
    private static final ParamsValueSetter<GBMParametersV3> PARAMS_SETTER =
            ParametersBuilderUtil.getParamSetters(GBMParametersV3.class);

    @Override
    protected GBMParametersV3 parseSpecificParams(final GBMParametersV3 h2oParams,
                                                  final Map<String, String> params,
                                                  final long randomSeed) {
        h2oParams.seed = randomSeed;
        params.forEach((paramName, value) -> cleanParam(value).ifPresent(paramValue ->
                PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue))
        );
        return h2oParams;
    }

    /**
     * Returns an empty representation of the algorithm specific parameters.
     *
     * @return An empty representation of the algorithm specific parameters.
     */
    @Override
    protected GBMParametersV3 getEmptyParams() {
        return new GBMParametersV3().fillFromImpl();
    }
}
