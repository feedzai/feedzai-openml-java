/*
 * Copyright 2020 Feedzai
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

package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.provider.descriptor.ModelParameter;
import com.feedzai.openml.provider.descriptor.fieldtype.BooleanFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.ModelParameterType;
import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;

import java.util.HashMap;
import java.util.Map;

/**
 * Class to return the LightGBM Algorithm's default parameters.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
public class TestParameters {

    /**
     * Gets default parameters.
     *
     * @return Returns all LightGBM ModelParameters with the default values.
     */
    public static Map<String, String> getDefaultParameters() {

        final Map<String, String> mapParams = new HashMap<>();
        for (final ModelParameter modelParameter : LightGBMDescriptorUtil.PARAMS) {
            final String defaultValue;
            final ModelParameterType type = modelParameter.getFieldType();

            if (type instanceof NumericFieldType) {
                final NumericFieldType numericType = (NumericFieldType) type;
                final double value = numericType.getDefaultValue();
                if (numericType.getParameterType() == NumericFieldType.ParameterConfigType.INT) {
                    defaultValue = String.valueOf((int) value);
                } else {
                    defaultValue = String.valueOf(value);
                }
            } else if (type instanceof ChoiceFieldType) {
                defaultValue = String.valueOf(((ChoiceFieldType) type).getDefaultValue());
            } else if (type instanceof BooleanFieldType) {
                defaultValue = String.valueOf(((BooleanFieldType) type).isDefaultTrue());
            } else {
                throw new RuntimeException("Invalid parameter type received.");
            }

            mapParams.put(modelParameter.getName(), defaultValue);
        }
        return mapParams;
    }
}
