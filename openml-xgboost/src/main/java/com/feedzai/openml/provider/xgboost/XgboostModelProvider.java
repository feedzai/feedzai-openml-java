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

import com.feedzai.openml.provider.MachineLearningProvider;
import com.feedzai.openml.provider.TrainingMachineLearningProvider;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import com.google.auto.service.AutoService;

import java.util.Optional;
import java.util.Set;

/**
 * Feedzai OpenML {@link MachineLearningProvider} for XGBoost, backed by the native {@code xgboost4j} JVM
 * package.
 *
 * <p>The provider is discovered by Pulse through the standard Java {@link java.util.ServiceLoader}
 * mechanism (via {@link AutoService}), so no changes to Pulse core are required to make it available -
 * only adding this module to the runtime classpath.
 *
 * @since 1.0.0
 */
@AutoService(MachineLearningProvider.class)
public class XgboostModelProvider implements TrainingMachineLearningProvider<XgboostModelCreator> {

    /**
     * The reported name of this provider.
     */
    public static final String PROVIDER_NAME = "XGBoost";

    @Override
    public String getName() {
        return PROVIDER_NAME;
    }

    @Override
    public Set<MLAlgorithmDescriptor> getAlgorithms() {
        return MLAlgorithmEnum.getDescriptors(XgboostAlgorithms.values());
    }

    @Override
    public Optional<XgboostModelCreator> getModelCreator(final String algorithmName) {
        return MLAlgorithmEnum.getByName(XgboostAlgorithms.values(), algorithmName)
                .map(algorithm -> new XgboostModelCreator());
    }
}
