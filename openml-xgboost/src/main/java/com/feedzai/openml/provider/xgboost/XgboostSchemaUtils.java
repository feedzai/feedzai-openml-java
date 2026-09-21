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

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;

/**
 * Shared helpers to turn Pulse {@link Instance}s into the flat {@code float[]} feature vectors XGBoost
 * expects.
 *
 * <p><b>Why this is shared between scoring and training:</b> XGBoost is purely numeric and positional -
 * it has no notion of feature names or categorical domains. Therefore the exact same column ordering and
 * encoding must be used when a model is trained and when it is later scored. Centralizing the feature
 * vector construction here guarantees that parity: both {@link XgboostModelCreator} (training) and
 * {@link XgboostClassificationModel} (scoring) build rows through this class.
 *
 * <p>Categorical fields arrive already encoded as {@code double} indices in the {@link Instance} (Pulse's
 * standard encoding), so they are copied as-is - identical to the LightGBM provider's behavior.
 *
 * @since 1.0.0
 */
final class XgboostSchemaUtils {

    /**
     * This class is not meant to be instantiated.
     */
    private XgboostSchemaUtils() {
    }

    /**
     * The number of predictive (non-target) features described by the schema.
     *
     * @param schema The dataset schema.
     * @return The number of predictive features.
     */
    static int numFeatures(final DatasetSchema schema) {
        return schema.getPredictiveFields().size();
    }

    /**
     * Builds the feature vector for a single {@link Instance}, in schema field order, skipping the target
     * field if one is present.
     *
     * @param instance The instance to convert.
     * @param schema   The dataset schema the instance conforms to.
     * @return A dense {@code float[]} with one entry per predictive feature.
     */
    static float[] featureRow(final Instance instance, final DatasetSchema schema) {
        final int numFields = schema.getFieldSchemas().size();
        final int targetIndex = schema.getTargetIndex().orElse(-1);
        final float[] row = new float[numFeatures(schema)];

        int featureIdx = 0;
        for (int fieldIdx = 0; fieldIdx < numFields; fieldIdx++) {
            if (fieldIdx == targetIndex) {
                continue;
            }
            row[featureIdx++] = (float) instance.getValue(fieldIdx);
        }
        return row;
    }
}
