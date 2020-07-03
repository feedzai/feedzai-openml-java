/*
 * Copyright 2020 Feedzai
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

package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.provider.descriptor.ModelParameter;
import com.feedzai.openml.provider.descriptor.fieldtype.BooleanFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;
import com.google.common.collect.ImmutableSet;

import java.util.Set;

/**
 * Utility to organize all the necessary Machine Learning Hyper-Parameters for configuring the training of LightGBM.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 */
public class LightGBMDescriptorUtil {

    /**
     * Boosting type parameter name.
     */
    public static final String BOOSTING_TYPE_PARAMETER_NAME = "boosting_type";

    /**
     * The number of boosting iterations parameter name.
     */
    public static final String NUM_ITERATIONS_PARAMETER_NAME = "num_iterations";

    /**
     * Bagging fraction parameter name.
     */
    public static final String BAGGING_FRACTION_PARAMETER_NAME = "bagging_fraction";

    /**
     * Bagging frequency parameter name.
     */
    public static final String BAGGING_FREQUENCY_PARAMETER_NAME = "bagging_freq";

    /**
     * Global seed parameter name.
     */
    public static final String SEED_PARAMETER_DESCRIPTION = "Seed";

    /**
     * Public-visible (UI) description of the bagging fraction parameter.
     */
    public static final String BAGGING_FRACTION_PARAMETER_DESCRIPTION = "Bagging fraction";

    /**
     * Public-visible (UI) description of the bagging frequency.
     */
    public static final String BAGGING_FREQUENCY_PARAMETER_DESCRIPTION = "Bagging frequency";

    /**
     * Public-visible (UI) description of the feature fraction parameter.
     */
    public static final String FEATURE_FRACTION_PARAMETER_DESCRIPTION = "Feature fraction by tree";

    /**
     * An alias to ease the readability of parameters' configuration that are not mandatory.
     */
    public static final boolean NOT_MANDATORY = false;

    /**
     * An alias to ease the readability of parameters' configuration that are not mandatory.
     */
    public static final boolean MANDATORY = true;

    /**
     * Helper method to return a range of type DOUBLE.
     *
     * @param minValue Minimum allowed value.
     * @param maxValue Maximum allowed value.
     * @param defaultValue Default value.
     * @return Double range with the specs above.
     */
    public static NumericFieldType doubleRange(final double minValue,
                                               final double maxValue,
                                               final double defaultValue) {
        return NumericFieldType.range(minValue, maxValue, NumericFieldType.ParameterConfigType.DOUBLE, defaultValue);
    }

    /**
     * Helper method to return a range of type INT.
     *
     * @param minValue Minimum allowed value.
     * @param maxValue Maximum allowed value.
     * @param defaultValue Default value.
     * @return Integer range with the specs above.
     */
    public static NumericFieldType intRange(final int minValue,
                                                   final int maxValue,
                                                   final int defaultValue) {
        return NumericFieldType.range(minValue, maxValue, NumericFieldType.ParameterConfigType.INT, defaultValue);
    }

    /**
     * Defines the set of model parameters accepted by the LightGBM model.
     */
    public static final Set<ModelParameter> PARAMS = ImmutableSet.of(
            // Core parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters
            new ModelParameter(
                    BOOSTING_TYPE_PARAMETER_NAME,
                    "Boosting type",
                    "Type of boosting model:\n"
                            + "'gbdt' is a good starting point,\n"
                            + "'goss' is faster but slightly less accurate,\n"
                            + "'dart' is much slower but might improve performance,\n"
                            + "'rf' is the random forest mode.",
                    MANDATORY,
                    new ChoiceFieldType(
                            ImmutableSet.of("gbdt", "rf", "dart", "goss"),
                            "gbdt"
                    )
            ),
            new ModelParameter(
                    NUM_ITERATIONS_PARAMETER_NAME,
                    "Number of booster iterations",
                    "LightGBM uses num_trees = num_classes * num_iterations for multi-classification.\n"
                            + "For binary classification it equals the number of trees.",
                    MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 100)
            ),
            new ModelParameter(
                    "learning_rate",
                    "Learning rate",
                    "Also named: 'shrinkage_rate' and 'eta'.\n"
                            + "In DART it also affects normalization weights of dropped trees.",
                    NOT_MANDATORY,
                    doubleRange(1E-99, Float.MAX_VALUE, 0.1)
            ),
            new ModelParameter(
                    "num_leaves",
                    "Maximum tree leaves",
                    "Maimum number of leaves per tree. Can be used to control over-fitting.",
                    NOT_MANDATORY,
                    intRange(1, 131072, 31)
            ),
            new ModelParameter(
                    "seed",
                    SEED_PARAMETER_DESCRIPTION,
                    "Seed used to generate other seeds. If other seeds are set, this one takes lower precedence.\n"
                            + " In LightGBM this is by default None, but in pulse it is by default 0.",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 0)
            ),
            // Learning Control Parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html#learning-control-parameters
            new ModelParameter(
                    "max_depth",
                    "Maximum tree depth",
                    "Limit the maximum depth for a tree.\n"
                            + "Used to control over-fitting when #data is small.\n"
                            + "The tree can still grow leaf-wise.\n"
                            + " <= 0 means no limit.",
                    NOT_MANDATORY,
                    intRange(-1, Integer.MAX_VALUE, -1)
            ),
            new ModelParameter(
                    "min_data_in_leaf",
                    "Minimum leaf data points",
                    "Minimum number of data points per leaf. Can be used to limit over-fitting.",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 20)
            ),
            new ModelParameter(
                    "min_sum_hessian_in_leaf",
                    "Minimum leaf hessian",
                    "Minimal sum of hessian per leaf. Can be used to limit over-fitting.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 1E-3)
            ),
            new ModelParameter(
                    BAGGING_FRACTION_PARAMETER_NAME,
                    BAGGING_FRACTION_PARAMETER_DESCRIPTION,
                    "Bagging sample fraction. Enabled if < 1.0.\n"
                            + "Used to randomly select part of the data without resampling.\n"
                            + "Can be used to speed up training and limit over-fitting.\n"
                            + "Requires '" + BAGGING_FREQUENCY_PARAMETER_DESCRIPTION + "' > 0.",
                    NOT_MANDATORY,
                    doubleRange(1E-20, 1.0, 1.0)
            ),
            new ModelParameter(
                    BAGGING_FREQUENCY_PARAMETER_NAME,
                    BAGGING_FREQUENCY_PARAMETER_DESCRIPTION,
                    "0 means no bagging, k means performing bagging at every k iteration.\n"
                            + "To enable bagging use '" + BAGGING_FRACTION_PARAMETER_DESCRIPTION + "' < 1.0.",
                    NOT_MANDATORY,
                    intRange(0, Integer.MAX_VALUE, 0)
            ),
            new ModelParameter(
                    "feature_fraction",
                    FEATURE_FRACTION_PARAMETER_DESCRIPTION,
                    "Sample fraction of features per iteration (tree) if smaller than 1.0.\n"
                            + "For example, if set to 0.8, 80% of features are selected before training a tree.\n"
                            + "Can speed up training and limit over-fitting.",
                    NOT_MANDATORY,
                    doubleRange(1E-4, 1.0, 1.0)
            ),
            new ModelParameter(
                    "feature_fraction_bynode",
                    "Feature fraction by tree node",
                    "Randomly select a fraction of the remaining tree's features per node \n"
                            + "(whether sampled after '"+FEATURE_FRACTION_PARAMETER_DESCRIPTION+"' or not).\n"
                            + "Active if smaller than 1.0.",
                    NOT_MANDATORY,
                    doubleRange(1E-4, 1.0, 1.0)
            ),
            /*
                TODO: PULSEDEV-31076 Support early stopping "early_stopping_round"
                      (https://lightgbm.readthedocs.io/en/latest/Parameters.html#early_stopping_round)
                      But can't implement without an in-train-validation dataset.
             */
            new ModelParameter(
                    "lambda_l1",
                    "Lambda L1",
                    "L1 regularization.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 0.0)
            ),
            new ModelParameter(
                    "lambda_l2",
                    "Lambda L2",
                    "L2 regularization.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 0.0)
            ),
            new ModelParameter(
                    "min_gain_to_split",
                    "Minimum split gain",
                    "Minimual gain to split a leaf.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 0.0)
            ),
            // DART parameters
            new ModelParameter(
                    "drop_rate",
                    "DART drop rate",
                    "DART's fraction of previous trees to drop during dropout.",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.1)
            ),
            new ModelParameter(
                    "max_drop",
                    "DART max drop",
                    "DART's maximum number of dropped trees during one boosting iteration.",
                    NOT_MANDATORY,
                    intRange(-1, Integer.MAX_VALUE, 50)
            ),
            new ModelParameter(
                    "skip_drop",
                    "DART skip drop",
                    "DART's probability of skipping the dropout procedure during a boosting iteration.",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.5)
            ),
            new ModelParameter(
                    "xgboost_dart_mode",
                    "DART XGBoost mode",
                    "Enable DART's XGboost mode.",
                    NOT_MANDATORY,
                    new BooleanFieldType(false)
            ),
            new ModelParameter(
                    "uniform_drop",
                    "DART uniform drop",
                    "Enable DART's uniform drop.",
                    NOT_MANDATORY,
                    new BooleanFieldType(false)
            ),
            // GOSS parameters
            new ModelParameter(
                    "top_rate",
                    "GOSS top retain rate",
                    "GOSS's retain ratio of large gradient data.",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.2)
            ),
            new ModelParameter(
                    "other_rate",
                    "GOSS other retain rate",
                    "GOSS's rate of retain for small gradient data.",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.1)
            ),
            // Generic parameters
            new ModelParameter(
                    "min_data_per_group",
                    "Minimum data per group",
                    "Minimum number of data points per categorical group.",
                    NOT_MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 100)
            ),
            new ModelParameter(
                    "max_cat_threshold",
                    "Maximum categories thresholds",
                    "Maximum number of thresholds to split a categorical feature",
                    NOT_MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 32)
            ),
            new ModelParameter(
                    "cat_l2",
                    "Categorical L2 regularization",
                    "L2 Regularization constant to use for the categorical splits.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 10.0)
            ),
            new ModelParameter(
                    "cat_smooth",
                    "Categorical smoothing",
                    "Reduce effect of noise in categorical feaures, specially for features with few data points.",
                    NOT_MANDATORY,
                    doubleRange(0.0, Double.MAX_VALUE, 10.0)
            ),
            new ModelParameter(
                    "max_cat_to_onehot",
                    "Max categories to one-hot encode",
                    "Maximum number of categories to which one-hot-encoding can be applied.",
                    NOT_MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 4)
            ),
            new ModelParameter(
                    "verbosity",
                    "Verbosity",
                    "Verbosity of the LightGBM backend. \n <0=Fatal, \n0=Error+Warnings, \n1=Info, \n>1 Debug",
                    NOT_MANDATORY,
                    intRange(Integer.MIN_VALUE, Integer.MAX_VALUE, 1)
            ),
            // IO Parameters
            // Dataset parameters
            new ModelParameter(
                    "max_bin",
                    "Max bins",
                    "Maximum number of bins that the feature variables will be bucketed in. Small number of bins may reduce training accuracy but increase generalization and thus reduce over-fitting. Up to 255, included, it will reduce memory consumption.",
                    NOT_MANDATORY,
                    intRange(2, Integer.MAX_VALUE, 255)
            ),
            new ModelParameter(
                    "min_data_in_bin",
                    "Minimium bin data",
                    "Minimum number of samples inside one bin. Limit over-fitting. E.g., not using 1 point per bin.",
                    NOT_MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 3)
            ),
            new ModelParameter(
                    "bin_construct_sample_cnt",
                    "Bin construction sample count",
                    "Number of sampled points to construct histogram bins. Setting to a larger value will give a better training result but increase data loading time. Set to a larger value if data is very sparse.",
                    NOT_MANDATORY,
                    intRange(1, Integer.MAX_VALUE, 200000)
            ),
            // TODO: https://lightgbm.readthedocs.io/en/latest/Parameters.html#is_enable_sparse ?? requires sparse CSR dataset?
            new ModelParameter(
                    "enable_bundle",
                    "Exclusive Feature Bundling (EFB)",
                    "Use Exclusive Feature Bundlinb (EFB). Disabling might slow down training for sparse datasets.",
                    NOT_MANDATORY,
                    new BooleanFieldType(true)
            ),
            new ModelParameter(
                    "zero_as_missing",
                    "Use zeroes as missing",
                    "If 'true' uses 0's as missing. Otherwise, missing/null values are treated as missing.",
                    NOT_MANDATORY,
                    new BooleanFieldType(false)
            ),
            new ModelParameter(
                    "is_unbalance",
                    "Unbalanced label",
                    "Set to true if training data is unbalanced. \nWhilst enabling this should increase the overall performance metric of the model, it will also result in poor estimates of the individual class probabilities. Cannot be used at the same time as 'scale_pos_weight'.", // TODO nam parameter in ui (scale_pos_weight)
                    MANDATORY,
                    new BooleanFieldType(false)
            )
            // TODO: https://lightgbm.readthedocs.io/en/latest/Parameters.html#scale_pos_weight ?? would require setting the pos label

    );
}
