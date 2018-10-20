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

import com.feedzai.openml.model.MachineLearningModel;
import com.feedzai.openml.provider.MachineLearningProvider;
import com.feedzai.openml.provider.TrainingMachineLearningProvider;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.google.auto.service.AutoService;

import java.util.Optional;
import java.util.Set;

/**
 * Implementation of the {@link MachineLearningProvider service provider}.
 * <p>
 * This class is a service responsible for providing {@link MachineLearningModel} that were generated in H2O.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
@AutoService(MachineLearningProvider.class)
public class H2OModelProvider implements TrainingMachineLearningProvider<H2OModelCreator>  {

    /**
     * Name of the service that provides {@link MachineLearningModel} generated in H2O.
     */
    public static final String H2O_NAME = "H2O";

    @Override
    public String getName() {
        return H2O_NAME;
    }

    @Override
    public Set<MLAlgorithmDescriptor> getAlgorithms() {
        return MLAlgorithmEnum.getDescriptors(H2OAlgorithm.values());
    }

    @Override
    public Optional<H2OModelCreator> getModelCreator(final String algorithmName) {

        return MLAlgorithmEnum.getByName(H2OAlgorithm.values(), algorithmName)
                .map(H2OAlgorithm::getAlgorithmDescriptor)
                .map(H2OModelCreator::new);
    }
}
