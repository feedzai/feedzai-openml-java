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

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.explanations.BaseExplanationsAlgorithm;

/**
 * A feature contribution provider for the predictions of the {@link LightGBMBinaryClassificationModel}.
 */
public class LightGBMExplanationsAlgorithm extends BaseExplanationsAlgorithm<LightGBMBinaryClassificationModel> {
    /**
     * Constructor.
     *
     * @param model The {@link LightGBMBinaryClassificationModel} which predictions will be explained.
     */
    public LightGBMExplanationsAlgorithm(final LightGBMBinaryClassificationModel model) {
        super(model);
    }

    @Override
    public double[] getFeatureContributions(final Instance instance) {
        return getModel().getFeatureContributions(instance);
    }
}
