/*
 * Copyright 2022 Feedzai
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
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.FreeTextFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;
import com.google.common.collect.ImmutableSet;

import com.google.common.collect.Sets;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Utility to organize all the necessary Machine Learning Hyper-Parameters for configuring the training of LightGBM.
 *
 * @author Andre Cruz (andre.cruz@feedzai.com)
 * @since 1.3.6
 */
public class FairGBMDescriptorUtil extends LightGBMDescriptorUtil {

    public static final String CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME = "constraint_group_column";

    /**
     * Defines the set of model parameters supported by the FairGBM algorithm.
     */
    static final Set<ModelParameter> PARAMS = Sets.union(ImmutableSet.of(
            // The single parameter that will change for every different dataset
            new ModelParameter(
                    CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME,
                    "(Fairness) Sensitive group column",
                    "Fairness constraints are enforced over this column.\n"
                            + "This column must be in categorical format.\n"
                            + "Start this string with `name:` to use the name of a column, \n"
                            + "e.g., `name:age_group` for a column named `age_group`.",
                    MANDATORY,
                    new FreeTextFieldType("")
//                    new FreeTextFieldType("", ".+")       # TODO: https://github.com/feedzai/feedzai-openml/issues/68
            ),

            new ModelParameter(
                    "constraint_type",
                    "(Fairness) Constraint type",
                    "Enforces group-wise parity on the given target metric for the selected group column. "
                            + "In general, FPR can be used for most detection settings "
                            + "to equalize the negative outcomes on legitimate individuals "
                            + "(false positives).",
                    NOT_MANDATORY,
                    new ChoiceFieldType(ImmutableSet.of("FPR", "FNR", "FPR,FNR"), "FPR")
            ),

            // Parameters related to global constraints
            new ModelParameter(
                    "global_constraint_type",
                    "(Fairness) Global constraint type",
                    "Type of global constraint to enforce during training of fairness constraints. "
                            + "For example, if you want to deploy your model on 5% FPR, use a global "
                            + "constraint on 5% FPR, so that the fairness constraints are trained by considering this "
                            + "FPR constraint in mind. Otherwise, fairness may not generalize well when you change "
                            + "the model's operating point.",
                    NOT_MANDATORY,
                    new ChoiceFieldType(ImmutableSet.of("FPR", "FNR", "FPR,FNR"), "FPR,FNR")
            ),
            new ModelParameter(
                    "global_target_fpr",
                    "(Fairness) Global target FPR",
                    "This is an inequality constraint: inactive when FPR is lower than the target. "
                            + "If you have an approximate value of FPR in mind for deploying the model, then "
                            + "set it here as well, so that fairness constraints can better adapt to such value. "
                            + "Oftentimes, some tension is required between global FPR and FNR constraints in order to "
                            + "achieve the target values (a global constraint on FPR and FNR simultaneously).",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.05)
            ),
            new ModelParameter(
                    "global_target_fnr",
                    "(Fairness) Global target FNR",
                    "This is an inequality constraint: inactive when FNR is lower than the target. "
                            + "If you have an approximate value of FNR in mind for deploying the model, then "
                            + "set it here as well, so that fairness constraints can better adapt to such value. "
                            + "Oftentimes, some tension is required between global FPR and FNR constraints in order to "
                            + "achieve the target values (a global constraint on FPR and FNR simultaneously).",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.5)
            ),

            new ModelParameter(
                    "objective",
                    "(Fairness) Objective function",
                    "For FairGBM you must use a constrained optimization function. "
                            + "`constrained_cross_entropy` is recommended for most cases.",
                    NOT_MANDATORY,
                    new ChoiceFieldType(
                            ImmutableSet.of("constrained_cross_entropy", "constrained_recall_objective"),
                            "constrained_cross_entropy")
            ),

            // Slack on the fairness constraints
            new ModelParameter(
                    "constraint_fpr_threshold",
                    "(Fairness) FPR slack for fairness",
                    "The slack when fulfilling fairness FPR constraints. "
                            + "The allowed difference between group-wise FPR. "
                            + "The value 0.0 enforces group-wise FPR to be *exactly* equal. "
                            + "Higher values lead to a less strict fairness enforcement.",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.0)
            ),
            new ModelParameter(
                    "constraint_fnr_threshold",
                    "(Fairness) FNR slack for fairness",
                    "The slack when fulfilling fairness FNR constraints. "
                            + "The allowed difference between group-wise FNR. "
                            + "The value 0.0 enforces group-wise FNR to be *exactly* equal. "
                            + "Higher values lead to a less strict fairness enforcement.",
                    NOT_MANDATORY,
                    doubleRange(0.0, 1.0, 0.0)
            ),

            // Eventually we want this parameter to not depend as much on the size of the dataset
            // But currently this needs to be changed for each dataset considering its size (larger for larger datasets)
            // See: https://github.com/feedzai/fairgbm/issues/7
            new ModelParameter(
                    "multiplier_learning_rate",
                    "(Fairness) Multipliers' learning rate",
                    "The Lagrangian multipliers control how strict the constraint enforcement is.",
                    NOT_MANDATORY,
                    NumericFieldType.min(Float.MIN_VALUE, NumericFieldType.ParameterConfigType.DOUBLE, 1e3)
            ), // NOTE: I'm using Float.MIN_VALUE here because the minimum value of a double in C++ depends on the architecture it's ran on, using float here is more conservative
            new ModelParameter(
                    "init_multipliers",
                    "(Fairness) Initial multipliers",
                    "The Lagrangian multipliers control how strict the constraint enforcement is. "
                            + "The default value is starting with zero `0` for each constraint.",
                    NOT_MANDATORY,
                    new FreeTextFieldType("")
//                    new FreeTextFieldType("", "^((\\d+(\\.\\d*)?,)*(\\d+(\\.\\d*)?))?$")  # TODO: https://github.com/feedzai/feedzai-openml/issues/68
            ),

            // These parameters probably shouldn't be changed in 90% of cases
            new ModelParameter(
                    "constraint_stepwise_proxy",
                    "(Fairness) Stepwise proxy for fairness constraints",
                    "The type of proxy function to use for the fairness constraint. "
                            + "We need to use a differentiable proxy function, as FPR and FNR have discontinuous gradients.",
                    NOT_MANDATORY,
                    new ChoiceFieldType(ImmutableSet.of("cross_entropy", "quadratic", "hinge"), "cross_entropy")
            ),
            new ModelParameter(
                    "objective_stepwise_proxy",
                    "(Fairness) Stepwise proxy for global constraints",
                    "The proxy function to use for the objective function. "
                            + "Only used when explicitly optimizing for Recall (or any other metric of the "
                            + "confusion matrix). Leave blank when using standard objectives, such as cross-entropy.",
                    NOT_MANDATORY,
                    new ChoiceFieldType(ImmutableSet.of("cross_entropy", "quadratic", "hinge", ""), "")
            ),

            // Override this parameter from LightGBM so we can disallow using RF
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
                            ImmutableSet.of("gbdt", "dart", "goss"),
                            "gbdt"
                    )
            )

            // TODO: assess whether these parameters would ever be useful
//            // These parameters probably shouldn't be changed in 99% of cases
//            new ModelParameter(
//                    "stepwise_proxy_margin",
//                    "",
//                    "",
//                    NOT_MANDATORY,
//                    new FreeTextFieldType("")
//            ),
//            new ModelParameter(
//                    "score_threshold",
//                    "",
//                    "",
//                    NOT_MANDATORY,
//                    new FreeTextFieldType("")
//            ),
//            new ModelParameter(
//                    "global_score_threshold",
//                    "",
//                    "",
//                    NOT_MANDATORY,
//                    new FreeTextFieldType("")
//            )

    ), LightGBMDescriptorUtil.PARAMS.stream()
                                    .filter(el -> !el.getName().equals(BOOSTING_TYPE_PARAMETER_NAME))
                                    .collect(Collectors.toSet()));

}
