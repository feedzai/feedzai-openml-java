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
import com.feedzai.openml.explanations.ModelExplainer;

/**
 * A feature contribution provider for the predictions of the {@link LightGBMBinaryClassificationModel}.
 */
public class LightGBMExplainer implements ModelExplainer<LightGBMBinaryClassificationModel> {
    /**
     * The {@link LightGBMBinaryClassificationModel} whose predictions will be explained.
     */
    private final LightGBMBinaryClassificationModel model;

    /**
     * Constructor.
     *
     * @param model The {@link LightGBMBinaryClassificationModel} whose predictions will be explained.
     */
    public LightGBMExplainer(final LightGBMBinaryClassificationModel model) {
        this.model = model;
    }

    @Override
    public double[] getFeatureContributions(final Instance instance) {
        // TODO move the logic somewhere so that we stop using the deprecated method
        return model.getFeatureContributions(instance);
    }
}
