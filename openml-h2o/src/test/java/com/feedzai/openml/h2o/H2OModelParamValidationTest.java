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
 * H2O Model Parameter validation test
 *
 * @author Antonio Silva (antonio.silva@feedzai.com)
 * @since 1.2.1
 */
@RunWith(Parameterized.class)
public class H2OModelParamValidationTest {

    private final H2OAlgorithm h2OAlgorithm;
    private final Map<String, String> defaultAlgoParams;
    private final String badParamName;
    private final String badParamValue;

    public H2OModelParamValidationTest(final H2OAlgorithm h2OAlgorithm,
                                       final Map<String, String> defaultAlgoParams,
                                       final String badParamName,
                                       final String badParamValue) {
        this.h2OAlgorithm = h2OAlgorithm;
        this.defaultAlgoParams = defaultAlgoParams;
        this.badParamName = badParamName;
        this.badParamValue = badParamValue;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                // { algorithm, default parameters, bad parameter name, bad parameter value }
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

        assertThat(validationErrors).isEmpty();
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
                .isNotEmpty()
                .allMatch(error -> error.getMessage().contains("Model has errors: "));
    }

    /**
     * Validates params for a given algorithm
     * @param params     The parameters to validate
     * @param algorithm  The H2O algorithm
     * @return           List of param validation errors
     *
     * @since 1.2.1
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