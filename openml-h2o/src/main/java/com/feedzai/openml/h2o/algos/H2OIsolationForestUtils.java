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

import com.feedzai.openml.h2o.params.ParametersBuilderUtil;
import com.feedzai.openml.h2o.params.ParamsValueSetter;
import com.feedzai.openml.provider.descriptor.ModelParameter;

import hex.schemas.IsolationForestV3.IsolationForestParametersV3;

import hex.tree.isofor.IsolationForest;
import java.util.Map;
import java.util.Set;

/**
 * Parameter inspector for H2O's isolation forest algorithm.
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 1.0.0
 */
public class H2OIsolationForestUtils extends AbstractUnsupervisedH2OAlgoUtils<IsolationForestParametersV3, IsolationForest> {

    /**
     * The set of parameters that are possible to define during the creation of an H2O Isolation forest model.
     */
    public static final Set<ModelParameter> PARAMETERS =
            ParametersBuilderUtil.getParametersFor(IsolationForestParametersV3.class, water.bindings.pojos.IsolationForestParametersV3.class);

    /**
     * The setter capable of assigning a value of a parameter to the right H2O REST POJO field.
     */
    private static final ParamsValueSetter<IsolationForestParametersV3> PARAMS_SETTER =
            ParametersBuilderUtil.getParamSetters(IsolationForestParametersV3.class);

    @Override
    protected IsolationForestParametersV3 parseSpecificParams(final IsolationForestParametersV3 h2oParams, final Map<String, String> params, final long randomSeed) {
        h2oParams.seed = randomSeed;
        params.forEach((paramName, value) -> cleanParam(value).ifPresent(paramValue ->
                PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue))
        );
        return h2oParams;
    }

    @Override
    protected IsolationForestParametersV3 getEmptyParams() {
        return new IsolationForestParametersV3().fillFromImpl();
    }

    @Override
    public IsolationForest getModel(final IsolationForestParametersV3 isolationForestParametersV3) {
        return new IsolationForest(isolationForestParametersV3.createAndFillImpl());
    }

}
