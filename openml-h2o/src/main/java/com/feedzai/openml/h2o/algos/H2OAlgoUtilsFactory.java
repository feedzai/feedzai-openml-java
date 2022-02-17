/*
 * Copyright 2021 Feedzai
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.h2o.H2OAlgorithm;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;

/**
 * Factory class responsible for providing the correct H20 Algorithm util.
 *
 * @author Antonio Silva (antonio.silva@feedzai.com)
 * @since 1.3.0
 */
public class H2OAlgoUtilsFactory {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OAlgoUtilsFactory.class);

    /**
     * Private constructor to avoid class instantiation.
     *
     */
    private H2OAlgoUtilsFactory() {
    }

    /**
     * Factory main method to get the correct H20 Algo Utils.
     *
     * @param algorithmDescriptor Descriptor for the algorithm we want the utils for.
     * @param schema              Datasource schema.
     * @return AbstractH2OAlgoUtils implementation for the algoDescriptor/schema provided.
     *
     */
    public static AbstractH2OAlgoUtils getH2OAlgoUtils(final MLAlgorithmDescriptor algorithmDescriptor,
                                                       final DatasetSchema schema) {
        switch (getH2OAlgorithm(algorithmDescriptor)) {
            case DISTRIBUTED_RANDOM_FOREST:
                return new H2ODrfUtils();
            case XG_BOOST:
                return new H2OXgboostUtils();
            case DEEP_LEARNING:
                return new H2ODeepLearningUtils();
            case GRADIENT_BOOSTING_MACHINE:
                return new H2OGbmUtils();
            case NAIVE_BAYES_CLASSIFIER:
                return new H2OBayesUtils();
            case GENERALIZED_LINEAR_MODEL:
                return new H2OGeneralizedLinearModelUtils(schema);
            case ISOLATION_FOREST:
                return new H2OIsolationForestUtils();
            default:
                final String errorMessage = String.format("Unsupported algorithm: %s", algorithmDescriptor.getAlgorithmName());
                logger.error(errorMessage);
                throw new IllegalArgumentException(errorMessage);
        }
    }

    /**
     * Resolves the H2O algorithm from the provided descriptor.
     *
     * @param algorithmDescriptor The algorithm descriptor from which the {@link H2OAlgorithm} is resolved.
     * @return The resolve {@link H2OAlgorithm}.
     *
     */
    private static H2OAlgorithm getH2OAlgorithm(final MLAlgorithmDescriptor algorithmDescriptor) {
        return MLAlgorithmEnum.getByName(H2OAlgorithm.values(), algorithmDescriptor.getAlgorithmName())
                              .orElseThrow(() -> new IllegalArgumentException("Unknown algorithm: " + algorithmDescriptor));
    }
}
