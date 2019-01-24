/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * © 2019 Feedzai, Strictly Confidential
 */

package com.feedzai.openml.h2o.algos;

import com.feedzai.openml.h2o.params.ParametersBuilderUtil;
import com.feedzai.openml.h2o.params.ParamsValueSetter;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import hex.schemas.IsolationForestV3.IsolationForestParametersV3;

import java.util.Map;
import java.util.Set;

/**
 * Parameter inspector for H2O's isolation forest algorithm.
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 1.0.0
 */
public class H2OIsolationForestUtils extends AbstractUnsupervisedH2OParamUtils<IsolationForestParametersV3> {

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
    IsolationForestParametersV3 parseSpecificParams(final IsolationForestParametersV3 h2oParams, final Map<String, String> params, final long randomSeed) {
        h2oParams.seed = randomSeed;
        params.forEach((paramName, value) -> cleanParam(value).ifPresent(paramValue ->
                PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue))
        );
        return h2oParams;
    }

    @Override
    IsolationForestParametersV3 getEmptyParams() {
        return new IsolationForestParametersV3().fillFromImpl();
    }

}
