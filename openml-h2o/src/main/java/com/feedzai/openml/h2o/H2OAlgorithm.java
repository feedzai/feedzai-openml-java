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

import com.feedzai.openml.h2o.algos.H2OBayesUtils;
import com.feedzai.openml.h2o.algos.H2ODeepLearningUtils;
import com.feedzai.openml.h2o.algos.H2ODrfUtils;
import com.feedzai.openml.h2o.algos.H2OGbmUtils;
import com.feedzai.openml.h2o.algos.H2OGeneralizedLinearModelUtils;
import com.feedzai.openml.h2o.algos.H2OXgboostUtils;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.MachineLearningAlgorithmType;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;

import static com.feedzai.openml.util.algorithm.MLAlgorithmEnum.createDescriptor;

/**
 * Specifies the algorithms of the models generated in H2O that are supported to be imported.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @see <a href="https://www.h2o.ai/">https://www.h2o.ai/</a> to find more information about the algorithms supported by
 * H2O and how to create models with those algorithms.
 * @since 0.1.0
 */
public enum H2OAlgorithm implements MLAlgorithmEnum {

    /**
     * Deep Learning is based on a multi-layer feed-forward artificial neural network that is trained with stochastic
     * gradient descent using back-propagation.
     */
    DEEP_LEARNING(createDescriptor(
            "Deep Learning",
            H2ODeepLearningUtils.PARAMETERS,
            MachineLearningAlgorithmType.MULTI_CLASSIFICATION,
            "http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html"
    )),

    /**
     * Distributed Random Forest (DRF) is a powerful classification and regression tool that when given a set of data, it
     * generates a forest of classification or regression trees, rather than a single classification or regression tree.
     */
    DISTRIBUTED_RANDOM_FOREST(createDescriptor(
            "Distributed Random Forest",
            H2ODrfUtils.PARAMETERS,
            MachineLearningAlgorithmType.MULTI_CLASSIFICATION,
            "http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html"
    )),

    /**
     * Gradient Boosting Machine (for Regression and Classification) is a forward learning ensemble method.
     */
    GRADIENT_BOOSTING_MACHINE(createDescriptor(
            "Gradient Boosting Machine",
            H2OGbmUtils.PARAMETERS,
            MachineLearningAlgorithmType.MULTI_CLASSIFICATION,
            "http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html"
    )),

    /**
     * Naive Bayes is a classification algorithm that relies on strong assumptions of the independence of covariates in
     * applying Bayes Theorem.
     */
    NAIVE_BAYES_CLASSIFIER(createDescriptor(
            "Naive Bayes Classifier",
            H2OBayesUtils.PARAMETERS,
            MachineLearningAlgorithmType.MULTI_CLASSIFICATION,
            "http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/naive-bayes.html"
    )),

    /**
     * XGBoost is a supervised learning algorithm that implements a process called boosting to yield accurate models.
     */
    XG_BOOST(createDescriptor(
            "XGBoost",
            H2OXgboostUtils.PARAMETERS,
            MachineLearningAlgorithmType.MULTI_CLASSIFICATION,
            "http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html"
    )),

    /**
     * Generalized Linear Models (GLM) estimate regression models for outcomes following exponential distributions.
     * <p>
     * We enforce it to always have a family parameter Binomial that causes it to be a classification algorithm.
     */
    GENERALIZED_LINEAR_MODEL(createDescriptor(
            "Generalized Linear Modeling",
            H2OGeneralizedLinearModelUtils.PARAMETERS,
            MachineLearningAlgorithmType.MULTI_CLASSIFICATION,
            "http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html"
    )),;

    /**
     * {@link MLAlgorithmDescriptor} for this algorithm.
     */
    public final MLAlgorithmDescriptor descriptor;

    /**
     * Constructor.
     *
     * @param descriptor The {@link MLAlgorithmDescriptor} for this algorithm.
     */
    H2OAlgorithm(final MLAlgorithmDescriptor descriptor) {
        this.descriptor = descriptor;
    }

    @Override
    public MLAlgorithmDescriptor getAlgorithmDescriptor() {
        return this.descriptor;
    }
}
