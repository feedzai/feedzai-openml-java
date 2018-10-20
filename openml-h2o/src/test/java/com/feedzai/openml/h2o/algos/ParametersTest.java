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

import com.feedzai.openml.h2o.H2OAlgorithm;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import org.junit.Test;

import java.util.Arrays;
import java.util.Set;

import static org.assertj.core.api.Java6Assertions.assertThat;


/**
 * Smoke test for H2O dynamic loading of model parameters.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 0.1.0
 */
public class ParametersTest {

    /**
     * Tests that all supported algorithms have at least some parameters as a kind of smoke test.
     */
    @Test
    public void testParametersExist() {

        Arrays.stream(H2OAlgorithm.values()).forEach(h2oAlg -> {
            final Set<ModelParameter> parameters = h2oAlg.getAlgorithmDescriptor().getParameters();

            assertThat(parameters)
                    .as("The parameters found")
                    .isNotEmpty();

            assertThat(parameters.size())
                    .as("The number of parameters")
                    .isGreaterThanOrEqualTo(10);

            assertThat(parameters.stream().anyMatch(ModelParameter::isMandatory))
                    .as("There is at least 1 mandatory parameter")
                    .isTrue();

            assertThat(parameters.stream().anyMatch(param -> !param.isMandatory()))
                    .as("There is at least 1 advanced parameter")
                    .isTrue();
        });
    }

}
