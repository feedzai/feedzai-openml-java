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

package com.feedzai.openml.h2o.params;


import water.api.schemas3.ModelParametersSchemaV3;

/**
 * Function capable of assigning a value of a parameter to the right corresponding H2O REST POJO field.
 *
 * @param <T> The concrete type of the ML Model params.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 0.1.0
 */
@FunctionalInterface
public interface ParamsValueSetter<T extends ModelParametersSchemaV3> {

    /**
     * Sets the given value in the given object's field corresponding to the given parameter name.
     *
     * @param paramsObj The object that holds the parameter values/fields.
     * @param paramName The name of the parameter.
     * @param paramVal  The intended value for the parameter.
     */
    void setValueIn(T paramsObj, String paramName, String paramVal);

}
