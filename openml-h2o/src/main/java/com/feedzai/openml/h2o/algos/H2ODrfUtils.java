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
import hex.schemas.DRFV3.DRFParametersV3;

import java.util.Map;
import java.util.Set;

/**
 * Utility class to hold relevant information to train H2O Distributed Random Forest models.
 *
 * @since 0.1.0
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 */
public final class H2ODrfUtils extends AbstractSupervisedH2OParamUtils<DRFParametersV3> {

    /**
     * The set of parameters that are possible to define during the creation of an H2O Distributed Random Forest model.
     */
    public static final Set<ModelParameter> PARAMETERS =
            ParametersBuilderUtil.getParametersFor(DRFParametersV3.class, water.bindings.pojos.DRFParametersV3.class);

    /**
     * The setter capable of assigning a value of a parameter to the right H2O REST POJO field.
     */
    private static final ParamsValueSetter<DRFParametersV3> PARAMS_SETTER =
            ParametersBuilderUtil.getParamSetters(DRFParametersV3.class);

    @Override
    public DRFParametersV3 parseSpecificParams(final DRFParametersV3 h2oParams, final Map<String, String> params, final long randomSeed) {
        h2oParams.seed = randomSeed;
        params.forEach((paramName, value) -> cleanParam(value).ifPresent(paramValue ->
                PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue))
        );
        return h2oParams;
    }

    @Override DRFParametersV3 getEmptyParams() {
        return new DRFParametersV3().fillFromImpl();
    }
}
