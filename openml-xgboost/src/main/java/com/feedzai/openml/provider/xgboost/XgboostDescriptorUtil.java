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

import com.feedzai.openml.provider.descriptor.ModelParameter;
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;
import com.google.common.collect.ImmutableSet;

import java.util.Set;

/**
 * Organizes the Machine Learning hyper-parameters exposed for training XGBoost models.
 *
 * <p>The parameter names match the native XGBoost parameter names (see
 * <a href="https://xgboost.readthedocs.io/en/stable/parameter.html">XGBoost Parameters</a>) so they can
 * be forwarded directly to the {@code xgboost4j} training API.
 *
 * @since 1.0.0
 */
final class XgboostDescriptorUtil {

    /**
     * Alias to ease readability of mandatory parameters.
     */
    private static final boolean MANDATORY = true;

    /**
     * Alias to ease readability of non-mandatory parameters.
     */
    private static final boolean NOT_MANDATORY = false;

    /**
     * The learning task and objective. Kept as a parameter (rather than hard-coded) so both binary and
     * multi-class objectives can be selected.
     */
    static final String OBJECTIVE_PARAMETER_NAME = "objective";

    /**
     * The number of boosting rounds (trees). Passed as the {@code nrounds} argument of
     * {@code XGBoost.train}, not as a booster parameter.
     */
    static final String NUM_ROUND_PARAMETER_NAME = "num_round";

    /**
     * Random seed parameter name.
     */
    static final String SEED_PARAMETER_NAME = "seed";

    /**
     * Number of parallel threads parameter name.
     */
    static final String NTHREAD_PARAMETER_NAME = "nthread";

    /**
     * The set of parameters accepted when training an XGBoost model.
     */
    static final Set<ModelParameter> PARAMS = ImmutableSet.of(
            new ModelParameter(
                    OBJECTIVE_PARAMETER_NAME,
                    "Objective",
                    "The learning task and corresponding objective:\n"
                            + "'binary:logistic' outputs the probability of the positive class,\n"
                            + "'binary:logitraw' outputs the raw (pre-sigmoid) score,\n"
                            + "'multi:softprob' outputs a per-class probability vector.",
                    MANDATORY,
                    new ChoiceFieldType(
                            ImmutableSet.of("binary:logistic", "binary:logitraw", "multi:softprob"),
                            "binary:logistic"
                    )
            ),
            new ModelParameter(
                    NUM_ROUND_PARAMETER_NAME,
                    "Number of boosting rounds",
                    "Number of boosting iterations (trees) to build.",
                    MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 100)
            ),
            new ModelParameter(
                    "eta",
                    "Learning rate (eta)",
                    "Step size shrinkage used in updates to prevent over-fitting. Also named 'learning_rate'.",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.3)
            ),
            new ModelParameter(
                    "max_depth",
                    "Maximum tree depth",
                    "Maximum depth of a tree. Increasing this value makes the model more complex and more\n"
                            + "likely to over-fit. 0 means no limit.",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 6)
            ),
            new ModelParameter(
                    "min_child_weight",
                    "Minimum child weight",
                    "Minimum sum of instance weight (hessian) needed in a child. Larger values are more\n"
                            + "conservative.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 1.0)
            ),
            new ModelParameter(
                    "gamma",
                    "Minimum split loss (gamma)",
                    "Minimum loss reduction required to make a further partition on a leaf node.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 0.0)
            ),
            new ModelParameter(
                    "subsample",
                    "Subsample ratio",
                    "Subsample ratio of the training instances. Setting it to 0.5 means XGBoost randomly\n"
                            + "samples half of the training data prior to growing trees.",
                    NOT_MANDATORY,
                    doubleRange(1E-6, 1.0, 1.0)
            ),
            new ModelParameter(
                    "colsample_bytree",
                    "Column subsample ratio by tree",
                    "Subsample ratio of columns when constructing each tree.",
                    NOT_MANDATORY,
                    doubleRange(1E-6, 1.0, 1.0)
            ),
            new ModelParameter(
                    "lambda",
                    "L2 regularization (lambda)",
                    "L2 regularization term on weights. Increasing this value makes the model more conservative.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 1.0)
            ),
            new ModelParameter(
                    "alpha",
                    "L1 regularization (alpha)",
                    "L1 regularization term on weights. Increasing this value makes the model more conservative.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 0.0)
            ),
            new ModelParameter(
                    SEED_PARAMETER_NAME,
                    "Seed",
                    "Random number seed used for reproducibility.",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 0)
            ),
            new ModelParameter(
                    NTHREAD_PARAMETER_NAME,
                    "Number of threads",
                    "Number of parallel threads used to run XGBoost. Defaults to 1 for deterministic behavior.",
                    NOT_MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 1)
            )
    );

    /**
     * This class is not meant to be instantiated.
     */
    private XgboostDescriptorUtil() {
    }

    /**
     * Helper that builds a {@code DOUBLE} numeric range.
     *
     * @param minValue     Minimum allowed value.
     * @param maxValue     Maximum allowed value.
     * @param defaultValue Default value.
     * @return The numeric field type.
     */
    private static NumericFieldType doubleRange(final double minValue,
                                                final double maxValue,
                                                final double defaultValue) {
        return NumericFieldType.range(minValue, maxValue, NumericFieldType.ParameterConfigType.DOUBLE, defaultValue);
    }

    /**
     * Helper that builds an {@code INT} numeric range.
     *
     * @param minValue     Minimum allowed value.
     * @param maxValue     Maximum allowed value.
     * @param defaultValue Default value.
     * @return The numeric field type.
     */
    private static NumericFieldType intRange(final int minValue,
                                             final int maxValue,
                                             final int defaultValue) {
        return NumericFieldType.range(minValue, maxValue, NumericFieldType.ParameterConfigType.INT, defaultValue);
    }
}
