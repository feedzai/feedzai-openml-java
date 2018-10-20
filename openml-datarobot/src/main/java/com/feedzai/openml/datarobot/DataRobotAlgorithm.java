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

package com.feedzai.openml.datarobot;

import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.MachineLearningAlgorithmType;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.google.common.collect.ImmutableSet;

import static com.feedzai.openml.util.algorithm.MLAlgorithmEnum.createDescriptor;

/**
 * Specifies the algorithms of the models generated in DataRobot that are supported to be imported.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @see <a href="https://www.datarobot.com/wiki/algorithm/">https://www.datarobot.com/wiki/algorithm/</a> to find more
 * information about the algorithms supported by DataRobot and how to create models with those algorithms.
 * @since 0.1.0
 */
public enum DataRobotAlgorithm implements MLAlgorithmEnum {

    /**
     * Generic binary classification algorithm.
     */
    GENERIC_BINARY_CLASSIFICATION(createDescriptor(
            "Generic Binary Classification",
            ImmutableSet.of(),
            MachineLearningAlgorithmType.BINARY_CLASSIFICATION,
            "https://www.datarobot.com/wiki/algorithm/"
    ));

    /**
     * {@link MLAlgorithmDescriptor} for this algorithm.
     */
    public final MLAlgorithmDescriptor descriptor;

    /**
     * Constructor.
     *
     * @param descriptor {@link MLAlgorithmDescriptor} for this algorithm.
     */
    DataRobotAlgorithm(final MLAlgorithmDescriptor descriptor) {
        this.descriptor = descriptor;
    }

    @Override
    public MLAlgorithmDescriptor getAlgorithmDescriptor() {
        return this.descriptor;
    }
}
