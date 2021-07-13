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

import hex.naivebayes.NaiveBayes;
import hex.schemas.NaiveBayesV3.NaiveBayesParametersV3;

import java.util.Map;
import java.util.Set;

/**
 * Utility class to hold relevant information to train H2O Naive Bayes models.
 *
 * @since 0.1.0
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 */
public final class H2OBayesUtils extends AbstractSupervisedH2OAlgoUtils<NaiveBayesParametersV3, NaiveBayes> {

    /**
     * The set of parameters that are possible to define during the creation of an H2O Bayes model.
     */
    public static final Set<ModelParameter> PARAMETERS =
            ParametersBuilderUtil.getParametersFor(NaiveBayesParametersV3.class, water.bindings.pojos.NaiveBayesParametersV3.class);

    /**
     * The setter capable of assigning a value of a parameter to the right H2O REST POJO field.
     */
    private static final ParamsValueSetter<NaiveBayesParametersV3> PARAMS_SETTER =
            ParametersBuilderUtil.getParamSetters(NaiveBayesParametersV3.class);

    @Override
    protected NaiveBayesParametersV3 parseSpecificParams(final NaiveBayesParametersV3 h2oParams,
                                                         final Map<String, String> params,
                                                         final long randomSeed) {
        h2oParams.seed = randomSeed;
        params.forEach((paramName, value) -> cleanParam(value).ifPresent(paramValue ->
                PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue))
        );
        return h2oParams;
    }

    @Override
    protected NaiveBayesParametersV3 getEmptyParams() {
        return new NaiveBayesParametersV3().fillFromImpl();
    }

    @Override
    public NaiveBayes getModel(final NaiveBayesParametersV3 naiveBayesParametersV3) {
        return new NaiveBayes(naiveBayesParametersV3.createAndFillImpl());
    }
}
