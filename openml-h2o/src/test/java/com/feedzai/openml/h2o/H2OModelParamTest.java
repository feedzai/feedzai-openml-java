/*
 * Copyright 2022 Feedzai
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

package com.feedzai.openml.h2o;

import com.feedzai.openml.h2o.algos.H2OBayesUtils;
import com.feedzai.openml.h2o.algos.H2ODeepLearningUtils;
import com.feedzai.openml.h2o.algos.H2ODrfUtils;
import com.feedzai.openml.h2o.algos.H2OGbmUtils;
import com.feedzai.openml.h2o.algos.H2OGeneralizedLinearModelUtils;
import com.feedzai.openml.h2o.algos.H2OIsolationForestUtils;
import com.feedzai.openml.h2o.algos.H2OXgboostUtils;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * H2O Model Parameter test. Validates if H2O model parameters match the model parameter names.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 */
@RunWith(Parameterized.class)
public class H2OModelParamTest {

    /**
     * Contains all model parameter names that can be defined in a given algorithm.
     */
    private final Collection<String> modelParameterNames;

    /**
     * Contains the default model parameters of a given algorithm.
     */
    private final Collection<ModelParameter> modelParameters;

    /**
     * Constructor.
     *
     * @param modelParameterNames All model parameter names.
     * @param modelParameters     Default model parameters.
     */
    public H2OModelParamTest(final Collection<String> modelParameterNames,
                             final Collection<ModelParameter> modelParameters) {
        this.modelParameterNames = modelParameterNames;
        this.modelParameters = modelParameters;
    }

    /**
     * Method that sets the parametrized data to use in the tests.
     *
     * @return data in the form of { all model parameter names, default parameters }.
     */
    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                { H2ODeepLearningUtils.PARAMETER_NAMES, H2ODeepLearningUtils.PARAMETERS },
                { H2ODrfUtils.PARAMETER_NAMES, H2ODrfUtils.PARAMETERS },
                { H2OGbmUtils.PARAMETER_NAMES, H2OGbmUtils.PARAMETERS },
                { H2OBayesUtils.PARAMETER_NAMES, H2OBayesUtils.PARAMETERS },
                { H2OXgboostUtils.PARAMETER_NAMES, H2OXgboostUtils.PARAMETERS },
                { H2OGeneralizedLinearModelUtils.PARAMETER_NAMES, H2OGeneralizedLinearModelUtils.PARAMETERS },
                { H2OIsolationForestUtils.PARAMETER_NAMES, H2OIsolationForestUtils.PARAMETERS }
        });
    }

    /**
     * Tests that the default H20 parameters are inside the complete lists of model parameters.
     */
    @Test
    public final void defaultParametersInsideCompleteList() {
        final Collection<String> defaultModelParameterNames = this.modelParameters.stream()
                .map(ModelParameter::getName)
                .collect(Collectors.toSet());

        assertThat(this.modelParameterNames)
                .as("Default parameters are contained inside the complete list of parameters")
                .containsAll(defaultModelParameterNames);
    }
}
