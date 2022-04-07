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

import hex.deeplearning.DeepLearning;
import hex.schemas.DeepLearningV3.DeepLearningParametersV3;
import org.apache.commons.lang3.StringUtils;

import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Utility class to hold relevant information to train H2O Deep Learning models.
 *
 * @since 0.1.0
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 */
public final class H2ODeepLearningUtils extends AbstractSupervisedH2OAlgoUtils<DeepLearningParametersV3, DeepLearning> {

    /**
     * H2O naming for hidden param.
     */
    public static final String HIDDEN = "hidden";

    /**
     * Separator used to split textual representation of list params.
     */
    private static final String LIST_PARAM_DELIMITER = ",";

    /**
     * The set of parameters that are possible to define during the creation of an H2O Deep Learning model.
     */
    public static final Set<ModelParameter> PARAMETERS =
            ParametersBuilderUtil.getParametersFor(DeepLearningParametersV3.class, water.bindings.pojos.DeepLearningParametersV3.class);

    /**
     * The complete collection of model parameter names of an H2O Deep Learning model.
     */
    public static final Set<String> PARAMETER_NAMES =
            ParametersBuilderUtil.getParametersNamesFor(water.bindings.pojos.DeepLearningParametersV3.class);

    /**
     * The setter capable of assigning a value of a parameter to the right H2O REST POJO field.
     */
    private static final ParamsValueSetter<DeepLearningParametersV3> PARAMS_SETTER =
            ParametersBuilderUtil.getParamSetters(DeepLearningParametersV3.class);

    @Override
    public DeepLearningParametersV3 parseSpecificParams(final DeepLearningParametersV3 h2oParams, final Map<String, String> params, final long randomSeed) {
        h2oParams.seed = randomSeed;
        params.forEach((paramName, value) -> {

            if (HIDDEN.equals(paramName)) {
                cleanParam(params.get(HIDDEN)).ifPresent(param ->
                        h2oParams.hidden = Stream.of(StringUtils.strip(param, "[]").split(LIST_PARAM_DELIMITER)).mapToInt(Integer::parseInt).toArray()
                );
                return;
            }

            cleanParam(value).ifPresent(paramValue -> PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue));
        });

        return h2oParams;
    }

    @Override
    protected DeepLearningParametersV3 getEmptyParams() {
        return new DeepLearningParametersV3().fillFromImpl();
    }

    @Override
    public DeepLearning getModel(final DeepLearningParametersV3 deepLearningParametersV3) {
        return new DeepLearning(deepLearningParametersV3.createAndFillImpl());
    }
}
