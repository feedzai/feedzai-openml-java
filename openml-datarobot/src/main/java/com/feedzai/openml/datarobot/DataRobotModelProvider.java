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

import com.feedzai.openml.model.MachineLearningModel;
import com.feedzai.openml.provider.MachineLearningProvider;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.google.auto.service.AutoService;

import java.util.Optional;
import java.util.Set;

/**
 * Implementation of the {@link MachineLearningProvider service provider}.
 * <p>
 * This class is a service responsible for providing {@link MachineLearningModel} that were generated in DataRobot.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
@AutoService(MachineLearningProvider.class)
public class DataRobotModelProvider implements MachineLearningProvider<DataRobotModelCreator>  {

    /**
     * Name of the service that provides {@link MachineLearningModel} generated in DataRobot.
     */
    public static final String PROVIDER_NAME = "DataRobot";

    @Override
    public String getName() {
        return PROVIDER_NAME;
    }

    @Override
    public Set<MLAlgorithmDescriptor> getAlgorithms() {
        return MLAlgorithmEnum.getDescriptors(DataRobotAlgorithm.values());
    }

    @Override
    public Optional<DataRobotModelCreator> getModelCreator(final String algorithmName) {
        return MLAlgorithmEnum.getByName(DataRobotAlgorithm.values(), algorithmName)
                .map(algorithm -> new DataRobotModelCreator());
    }
}
