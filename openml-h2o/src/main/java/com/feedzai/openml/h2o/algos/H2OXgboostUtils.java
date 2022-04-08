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
import hex.schemas.XGBoostV3.XGBoostParametersV3;

import java.util.Map;
import java.util.Set;

/**
 * Utility class to hold relevant information to train H2O XGBoost models.
 *
 * @since 0.1.0
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 */
public final class H2OXgboostUtils extends AbstractSupervisedH2OParamUtils<XGBoostParametersV3> {

    /**
     * The set of parameters that are possible to define during the creation of an H2O XGBoost model.
     */
    public static final Set<ModelParameter> PARAMETERS =
            ParametersBuilderUtil.getParametersFor(XGBoostParametersV3.class, water.bindings.pojos.XGBoostParametersV3.class);

    /**
     * The complete collection of model parameter names of an H2O XGBoost model.
     */
    public static final Set<String> PARAMETER_NAMES =
            ParametersBuilderUtil.getAllParametersNamesFor(water.bindings.pojos.XGBoostParametersV3.class);

    /**
     * The setter capable of assigning a value of a parameter to the right H2O REST POJO field.
     */
    private static final ParamsValueSetter<XGBoostParametersV3> PARAMS_SETTER =
            ParametersBuilderUtil.getParamSetters(XGBoostParametersV3.class);

    @Override
    protected XGBoostParametersV3 parseSpecificParams(final XGBoostParametersV3 h2oParams,
                                                      final Map<String, String> params,
                                                      final long randomSeed) {

        h2oParams.seed = randomSeed;
        params.forEach((paramName, value) -> cleanParam(value).ifPresent(paramValue ->
                PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue))
        );
        return h2oParams;
    }

    @Override
    protected XGBoostParametersV3 getEmptyParams() {
        return new XGBoostParametersV3().fillFromImpl();
    }
}
