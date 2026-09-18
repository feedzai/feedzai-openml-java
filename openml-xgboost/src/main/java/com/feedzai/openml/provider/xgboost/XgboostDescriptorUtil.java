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
     * The learning task and objective. Restricted to {@code binary:logistic} - the only objective whose
     * single probability output maps cleanly onto the binary {@code [1 - p, p]} class distribution this
     * provider produces.
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
                    "The learning task and corresponding objective. Only 'binary:logistic' is supported:\n"
                            + "it outputs the probability of the positive class, which this provider maps\n"
                            + "directly onto a [1 - p, p] class distribution.",
                    MANDATORY,
                    new ChoiceFieldType(
                            ImmutableSet.of("binary:logistic"),
                            "binary:logistic"
                    )
            ),
            new ModelParameter(
                    NUM_ROUND_PARAMETER_NAME,
                    "Number of boosting rounds",
                    "Number of boosting iterations (trees) to build.",
                    MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 50)
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
                    "max_delta_step",
                    "Maximum delta step",
                    "Maximum delta step allowed for each leaf output. 0 means no constraint. Can help make\n"
                            + "the update more conservative, notably with very imbalanced classes.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 0.0)
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
                    "colsample_bylevel",
                    "Column subsample ratio by level",
                    "Subsample ratio of columns for each level. Subsampling occurs once for every new depth\n"
                            + "level reached in a tree, over the columns already chosen for the current tree.",
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
                    "scale_pos_weight",
                    "Scale positive weight",
                    "Controls the balance of positive and negative weights, useful for imbalanced classes.\n"
                            + "A typical value to consider is sum(negative instances) / sum(positive instances).",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 1.0)
            ),
            new ModelParameter(
                    "tree_method",
                    "Tree method",
                    "The tree construction algorithm used in XGBoost:\n"
                            + "'auto' (same as 'hist'), 'exact', 'approx' and 'hist'.",
                    NOT_MANDATORY,
                    new ChoiceFieldType(
                            ImmutableSet.of("auto", "exact", "approx", "hist"),
                            "auto"
                    )
            ),
            new ModelParameter(
                    "grow_policy",
                    "Grow policy",
                    "Controls the way new nodes are added to the tree:\n"
                            + "'depthwise' splits at nodes closest to the root, 'lossguide' splits at nodes with\n"
                            + "the highest loss change.",
                    NOT_MANDATORY,
                    new ChoiceFieldType(
                            ImmutableSet.of("depthwise", "lossguide"),
                            "depthwise"
                    )
            ),
            new ModelParameter(
                    "max_leaves",
                    "Maximum number of leaves",
                    "Maximum number of nodes to be added. 0 means no limit. Not used by the 'exact' tree\n"
                            + "method.",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 0)
            ),
            new ModelParameter(
                    "max_bin",
                    "Maximum number of bins",
                    "Maximum number of discrete bins used to bucket continuous features. Only used by the\n"
                            + "'hist' and 'approx' tree methods. Increasing it improves optimality of splits at the\n"
                            + "cost of higher computation time.",
                    NOT_MANDATORY,
                    intRange(2, Integer.MAX_VALUE, 256)
            ),
            new ModelParameter(
                    SEED_PARAMETER_NAME,
                    "Seed",
                    "Random number seed used for reproducibility. Note that reproducibility is only fully\n"
                            + "guaranteed with 'nthread' = 1: with multiple threads the 'hist' tree method's\n"
                            + "reduction order is non-deterministic, so results may vary slightly run to run even\n"
                            + "with a fixed seed.",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 0)
            ),
            new ModelParameter(
                    NTHREAD_PARAMETER_NAME,
                    "Number of threads",
                    "Number of parallel threads used to run XGBoost. 0 (the default) uses all available\n"
                            + "threads, matching the behavior of the previous H2O-backed implementation. Values other\n"
                            + "than 1 make training non-deterministic even with a fixed 'seed' (see the seed\n"
                            + "parameter).",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 0)
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
