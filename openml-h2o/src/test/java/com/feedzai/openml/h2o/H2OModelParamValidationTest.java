/*
 * Copyright 2021 Feedzai
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

import com.google.common.io.Files;
import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * H2O Model Parameter validation test. Validates if H2O model parameters are being properly validated.
 *
 * @author Antonio Silva (antonio.silva@feedzai.com)
 * @since 1.2.1
 */
@RunWith(Parameterized.class)
public class H2OModelParamValidationTest {

    /**
     * The algorithm we want to check the params of.
     */
    private final H2OAlgorithm h2OAlgorithm;

    /**
     * The default set of params for this algorithm.
     */
    private final Map<String, String> defaultAlgoParams;

    /**
     * The name of the param we are going to change to check the validation.
     */
    private final String badParamName;

    /**
     * The value to put in the param we are manipulating.
     */
    private final String badParamValue;

    /**
     * Test class constructor used to inject parametrized values into the fields.
     *
     * @param h2OAlgorithm      The algorithm we want to check the params of.
     * @param defaultAlgoParams The default set of params for this algorithm.
     * @param badParamName      The name of the param we are going to change to check the validation.
     * @param badParamValue     The value to put in the param we are manipulating.
     */
    public H2OModelParamValidationTest(final H2OAlgorithm h2OAlgorithm,
                                       final Map<String, String> defaultAlgoParams,
                                       final String badParamName,
                                       final String badParamValue) {
        this.h2OAlgorithm = h2OAlgorithm;
        this.defaultAlgoParams = defaultAlgoParams;
        this.badParamName = badParamName;
        this.badParamValue = badParamValue;
    }

    /**
     * Method that sets the parametrized data to use in the tests.
     *
     * @return data in the form of { algorithm, default parameters, bad parameter name, bad parameter value }.
     */
    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                { H2OAlgorithm.NAIVE_BAYES_CLASSIFIER, H2OAlgorithmTestParams.getBayes(), "laplace", "-0.25" },
                { H2OAlgorithm.ISOLATION_FOREST, H2OAlgorithmTestParams.getIsolationForest(), "mtries", "-3" },
                { H2OAlgorithm.XG_BOOST, H2OAlgorithmTestParams.getXgboost(), "learn_rate", "2" },
                { H2OAlgorithm.DEEP_LEARNING, H2OAlgorithmTestParams.getDeepLearning(), "mini_batch_size", "0" },
                { H2OAlgorithm.DISTRIBUTED_RANDOM_FOREST, H2OAlgorithmTestParams.getDrf(), "mtries", "-3" },
                { H2OAlgorithm.GRADIENT_BOOSTING_MACHINE, H2OAlgorithmTestParams.getGbm(), "max_abs_leafnode_pred", "0" },
                { H2OAlgorithm.GENERALIZED_LINEAR_MODEL, H2OAlgorithmTestParams.getGlm(), "alpha", "2" }
        });
    }


    /**
     * Tests H20 parameter validation when all parameters correct.
     */
    @Test
    public final void testValidationOKWhenAllParametersAreValid() {
        final List<ParamValidationError> validationErrors = validateParamsForAlgorithm(defaultAlgoParams, h2OAlgorithm);

        assertThat(validationErrors)
                .as("There should be no validation errors when the params are all ok")
                .isEmpty();
    }

    /**
     * Tests H20 parameter validation for with one parameter invalid.
     */
    @Test
    public final void testHasValidationErrorsWhenParametersAreInvalid() {
        final Map<String, String> params = new HashMap<>(defaultAlgoParams);
        params.put(badParamName, badParamValue);

        final List<ParamValidationError> validationErrors = validateParamsForAlgorithm(params, h2OAlgorithm);

        assertThat(validationErrors)
                .as("There should be validation because there is an invalid parameter")
                .isNotEmpty()
                .allMatch(error -> error.getMessage().contains("Model has errors: "));
    }

    /**
     * Validates params for a given algorithm.
     *
     * @param params     The parameters to validate.
     * @param algorithm  The H2O algorithm.
     *
     * @return List of param validation errors.
     */
    private List<ParamValidationError> validateParamsForAlgorithm(final Map<String, String> params,
                                                                 final MLAlgorithmEnum algorithm) {
        final Optional<H2OModelCreator> modelCreatorOptional = new H2OModelProvider().getModelCreator(algorithm.getName());
        assertThat(modelCreatorOptional)
                .as("Model loader must be present")
                .isPresent();

        final H2OModelCreator loader = modelCreatorOptional.get();

        final File tempDir = Files.createTempDir();
        tempDir.deleteOnExit();

        return loader.validateForFit(tempDir.toPath(), H2ODatasetMixin.SCHEMA, params);
    }

}