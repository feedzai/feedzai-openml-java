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
package com.feedzai.openml.java.utils;

import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import com.feedzai.openml.provider.descriptor.fieldtype.BooleanFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.FreeTextFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import mockit.Expectations;
import mockit.Mocked;
import mockit.integration.junit4.JMockit;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Validates the behaviour of {@link ModelParameterUtils}.
 */
@RunWith(JMockit.class)
public class ModelParameterUtilsTest {

    /**
     * Validates that effective model parameters are being correctly calculated.s
     *
     * @param mlAlgorithmDescriptor The mocked ML algorithm.
     */
    @Test
    public final void effectiveModelParameterValues(@Mocked final MLAlgorithmDescriptor mlAlgorithmDescriptor) {

        final Map<String, String> expectedParams = ImmutableMap.of(
                "param0", "false",
                "param2", "2.0",
                "param3", "3",
                "param4", "53",
                "param1", "99"
        );

        new Expectations() {{
            mlAlgorithmDescriptor.getParameters();
            result = ImmutableSet.of(
                    new ModelParameter("param0", "", "", true, new BooleanFieldType(false)),
                    new ModelParameter("param1", "", "", true, new FreeTextFieldType("1")),
                    new ModelParameter("param2", "", "", true, NumericFieldType.min(0d, NumericFieldType.ParameterConfigType.INT, 2d)),
                    new ModelParameter("param3", "", "", true, new ChoiceFieldType(ImmutableSet.of("1", "2", "3"), "3"))
            );
        }};

        final Map<String, String> effectiveParams = ModelParameterUtils.getEffectiveModelParameterValues(
                mlAlgorithmDescriptor,
                ImmutableSet.of("param0", "param1", "param2", "param3", "param4"),
                ImmutableMap.of(
                        "param0", "false",
                        "param1", "99",
                        "param4", "53",
                        "param5", "100"
                )
        );

        assertThat(effectiveParams)
                .as("effective model parameter values")
                .isEqualTo(expectedParams);
    }
}
