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
import com.feedzai.openml.provider.descriptor.fieldtype.ModelParameterType;
import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Contains utility methods to manipulate {@link ModelParameter}. This allows to retrieve the default values used for
 * each parameter.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 */
public final class ModelParameterUtils {

    /**
     * Constructor of the utility class.
     */
    private ModelParameterUtils() {
    }

    /**
     * Retrieves the effective model parameters to be used by an OpenML provider. The effective instance is retrieved
     * after getting the values of the default model parameters, then it is merged with the new model parameters.
     *
     * @param algorithm      The description of a Machine Learning algorithm.
     * @param parameterNames The complete list of model parameter names.
     * @param newParams      The collection of new model parameters.
     * @return The collection of effective model parameters.
     */
    public static Map<String, String> getEffectiveModelParameterValues(final MLAlgorithmDescriptor algorithm,
                                                                       final Set<String> parameterNames,
                                                                       final Map<String, String> newParams) {

        final Map<String, String> defaultValues = getDefaultModelParameterValues(algorithm);

        final Map<String, String> effectiveModelParameter = new HashMap<>();

        defaultValues.entrySet().stream()
                .filter(entry -> !newParams.containsKey(entry.getKey()))
                .forEach(entry -> effectiveModelParameter.put(entry.getKey(), entry.getValue()));
        parameterNames.stream()
                .filter(newParams::containsKey)
                .forEach(parameter -> effectiveModelParameter.put(parameter, newParams.get(parameter)));

        return effectiveModelParameter;
    }

    /**
     * Retrieves the default model parameters from a {@link MLAlgorithmDescriptor}.
     *
     * @param algorithm The description of a Machine Learning algorithm.
     * @return The collection of default model parameters.
     */
    private static Map<String, String> getDefaultModelParameterValues(final MLAlgorithmDescriptor algorithm) {

        return algorithm.getParameters().stream()
                .collect(Collectors.toMap(
                        ModelParameter::getName,
                        parameter -> getModelParameterTypeDefaultValue(parameter.getFieldType())
                ));
    }

    /**
     * Retrieves the default value from a {@link ModelParameterType}.
     *
     * @param fieldType The description of a model parameter field.
     * @return The default value of a model parameter.
     */
    private static String getModelParameterTypeDefaultValue(final ModelParameterType fieldType) {

        if (fieldType instanceof FreeTextFieldType) {
            final FreeTextFieldType freeTextFieldType = (FreeTextFieldType) fieldType;
            return freeTextFieldType.getDefaultValue();

        } else if (fieldType instanceof NumericFieldType) {
            final NumericFieldType numericFieldType = (NumericFieldType) fieldType;
            return String.valueOf(numericFieldType.getDefaultValue());

        } else if (fieldType instanceof ChoiceFieldType) {
            final ChoiceFieldType choiceFieldType = (ChoiceFieldType) fieldType;
            return choiceFieldType.getDefaultValue();

        } else if (fieldType instanceof BooleanFieldType) {
            final BooleanFieldType booleanFieldType = (BooleanFieldType) fieldType;
            return String.valueOf(booleanFieldType.isDefaultTrue());
        }

        throw new IllegalArgumentException(String.format("Unrecognized model parameter type [%s]", fieldType));
    }
}
