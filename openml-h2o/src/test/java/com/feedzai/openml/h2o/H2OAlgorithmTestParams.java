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

import com.google.common.collect.ImmutableMap;
import hex.deepwater.DeepWaterParameters;
import hex.glm.GLMModel;

import java.util.Map;

/**
 * Test parameters to use in H2O Train tests.
 *
 * @author Luis Reis (luis.reis@feedzai.com)
 * @since 0.1.0
 */
public final class H2OAlgorithmTestParams {

    /**
     * Private constructor for util class.
     */
    private H2OAlgorithmTestParams() {
    }

    /**
     * Test parameters used to train {@link H2OAlgorithm#NAIVE_BAYES_CLASSIFIER} models.
     *
     * @return Map of test parameter to value.
     */
    public static Map<String, String> getBayes() {
        return ImmutableMap.<String, String>builder()
                .put("laplace", "0.25")
                .put("min_sdev", "0.002")
                .put("eps_sdev", "0")
                .put("min_prob", "0.003")
                .put("eps_prob", "0")
                .put("compute_metrics", "false")
                .build();
    }

    /**
     * Test parameters used to train {@link H2OAlgorithm#DEEP_LEARNING} models.
     *
     * @return Map of test parameter to value.
     */
    public static Map<String, String> getDeepLearning() {
        return ImmutableMap.<String, String>builder()
                .put("activation", DeepWaterParameters.Activation.Rectifier.toString())
                .put("epochs", "7")
                .put("hidden", "200,150,100")
                .put("variable_importances", "false")
                .build();
    }

    /**
     * Test parameters used to train {@link H2OAlgorithm#DISTRIBUTED_RANDOM_FOREST} models.
     *
     * @return Map of test parameter to value.
     */
    public static Map<String, String> getDrf() {
        return ImmutableMap.<String, String>builder()
                .put("ntrees", "51")
                .put("max_depth", "5")
                .put("min_rows", "10")
                .put("nbins", "20")
                .put("sample_rate", "1")
                .put("mtries", "-1")
                .build();
    }

    /**
     * Test parameters used to train {@link H2OAlgorithm#GRADIENT_BOOSTING_MACHINE} models.
     *
     * @return Map of test parameter to value.
     */
    public static Map<String, String> getGbm() {
        return ImmutableMap.<String, String>builder()
                .put("ntrees", "49")
                .put("max_depth", "6")
                .put("min_rows", "1")
                .put("nbins", "21")
                .put("learn_rate", "0.3")
                .put("sample_rate", "1")
                .put("col_sample_rate", "1")
                .build();
    }

    /**
     * Test parameters used to train {@link H2OAlgorithm#XG_BOOST} models.
     *
     * @return Map of test parameter to value.
     */
    public static Map<String, String> getXgboost() {
        return ImmutableMap.<String, String>builder()
                .put("ntrees", "151")
                .put("max_depth", "10")
                .put("min_rows", "1")
                .put("min_child_weight", "1")
                .put("learn_rate", "0.35")
                .put("eta", "0.3")
                .put("sample_rate", "1")
                .put("subsample", "1")
                .put("col_sample_rate", "1")
                .put("colsample_bylevel", "1")
                .build();
    }

    /**
     * Test parameters used to train {@link H2OAlgorithm#GENERALIZED_LINEAR_MODEL} models.
     *
     * @return Map of test parameter to value.
     */
    public static Map<String, String> getGlm() {
        return ImmutableMap.<String, String>builder()
                .put("solver", GLMModel.GLMParameters.Solver.AUTO.toString())
                .put("alpha", "0.5")
                .put("lambda", "1")
                .put("lambda_search", "true")
                .put("standardize", "true")
                .put("non_negative", "true")
                .put("obj_reg", "-1")
                .build();

    }
}
