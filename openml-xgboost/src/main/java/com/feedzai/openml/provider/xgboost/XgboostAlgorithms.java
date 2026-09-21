/*
 * Copyright 2026 Feedzai
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

package com.feedzai.openml.provider.xgboost;

import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.MachineLearningAlgorithmType;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;

import static com.feedzai.openml.util.algorithm.MLAlgorithmEnum.createDescriptor;

/**
 * Specifies the XGBoost algorithms that can be imported and trained through this provider.
 *
 * @since 1.0.0
 */
public enum XgboostAlgorithms implements MLAlgorithmEnum {

    /**
     * XGBoost binary classifier.
     */
    XGBOOST_BINARY_CLASSIFIER(createDescriptor(
            "DMLC - XGBoost",
            XgboostDescriptorUtil.PARAMS,
            MachineLearningAlgorithmType.SUPERVISED_BINARY_CLASSIFICATION,
            "https://xgboost.readthedocs.io/"
    ));

    /**
     * {@link MLAlgorithmDescriptor} for this algorithm.
     */
    private final MLAlgorithmDescriptor descriptor;

    /**
     * Constructor.
     *
     * @param descriptor {@link MLAlgorithmDescriptor} for this algorithm.
     */
    XgboostAlgorithms(final MLAlgorithmDescriptor descriptor) {
        this.descriptor = descriptor;
    }

    @Override
    public MLAlgorithmDescriptor getAlgorithmDescriptor() {
        return this.descriptor;
    }
}
