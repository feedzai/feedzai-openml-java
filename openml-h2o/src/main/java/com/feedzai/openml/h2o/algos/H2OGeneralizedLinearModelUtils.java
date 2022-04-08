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

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.h2o.params.ParametersBuilderUtil;
import com.feedzai.openml.h2o.params.ParamsValueSetter;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import hex.glm.GLMModel;
import hex.schemas.GLMV3.GLMParametersV3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Utility class to hold relevant information to train H2O Generalized Linear Models.
 *
 * @since 0.1.0
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 */
public final class H2OGeneralizedLinearModelUtils extends AbstractSupervisedH2OParamUtils<GLMParametersV3> {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OGeneralizedLinearModelUtils.class);

    /**
     * A "private" parameter that we always configure accordingly to make sure that the algorithm is for classification
     * problems.
     */
    private static final String FAMILY = "family";

    /**
     * The set of parameters that are possible to define during the creation of an H2O GLM model.
     */
    public static final Set<ModelParameter> PARAMETERS =
            ParametersBuilderUtil.getParametersFor(GLMParametersV3.class, water.bindings.pojos.GLMParametersV3.class)
                    .stream()
                    // Excluded since we "hard-code" to classification.
                    .filter(modelParam -> !FAMILY.equals(modelParam.getName()))
                    // H2O UI excludes these in a hard-coded way...so do we.
                    .filter(modelParameter -> !"tweedie_link_power".equals(modelParameter.getName()) &&
                            !"tweedie_variance_power".equals(modelParameter.getName()) &&
                            !"nlambdas".equals(modelParameter.getName()) && !"early_stopping".equals(modelParameter.getName())
                    )
                    .collect(Collectors.toSet());

    /**
     * The complete collection of model parameter names of an H2O GLM model.
     */
    public static final Set<String> PARAMETER_NAMES =
            ParametersBuilderUtil.getAllParametersNamesFor(water.bindings.pojos.GLMParametersV3.class);

    /**
     * The setter capable of assigning a value of a parameter to the right H2O REST POJO field.
     */
    private static final ParamsValueSetter<GLMParametersV3> PARAMS_SETTER =
            ParametersBuilderUtil.getParamSetters(GLMParametersV3.class);

    /**
     * The number of classes to predict.
     */
    private final int numberClasses;

    /**
     * Creates the utility bound to a target field.
     *
     * @param schema The target field.
     */
    public H2OGeneralizedLinearModelUtils(final DatasetSchema schema) {
        this.numberClasses = schema.getTargetFieldSchema()
                .map(FieldSchema::getValueSchema)
                .map(CategoricalValueSchema.class::cast)
                .map(CategoricalValueSchema::getNominalValues)
                .map(Collection::size)
                .orElseThrow(() -> new IllegalArgumentException("Generalized Linear Models require a target field, which is not preset in the schema provided."));
    }

    @Override
    public GLMParametersV3 parseSpecificParams(final GLMParametersV3 h2oParams, final Map<String, String> params, final long randomSeed) {
        h2oParams.seed = randomSeed;
        h2oParams.family = this.numberClasses > 2 ? GLMModel.GLMParameters.Family.multinomial : GLMModel.GLMParameters.Family.binomial;
        params.forEach((paramName, value) -> {

            if (FAMILY.equals(paramName)) {
                logger.warn("Ignoring configured parameter " + paramName + " with value " + value + " since that parameter is not allowed.");
                return;
            }

            cleanParam(value).ifPresent(paramValue -> PARAMS_SETTER.setValueIn(h2oParams, paramName, paramValue));
        });
        return h2oParams;
    }

    @Override
    protected GLMParametersV3 getEmptyParams() {
        return new GLMParametersV3().fillFromImpl();
    }
}
