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
import com.feedzai.openml.model.ClassificationMLModel;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/**
 * A classification model backed by a native XGBoost {@link Booster}, used for real-time single-instance
 * scoring.
 *
 * <p>Scoring uses {@link Booster#inplace_predict(float[], int, int, float)} on a single-row feature
 * vector, which avoids allocating a {@code DMatrix} per prediction. The native booster handle is not
 * thread-safe, so predictions are serialized on a private lock (mirrors the H2O provider's approach).
 *
 * @since 1.0.0
 */
public class XgboostClassificationModel implements ClassificationMLModel {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(XgboostClassificationModel.class);

    /**
     * Value used to signal a missing feature to XGBoost.
     */
    private static final float MISSING_VALUE = Float.NaN;

    /**
     * The native XGBoost booster.
     */
    private final Booster booster;

    /**
     * The schema the model uses.
     */
    private final DatasetSchema schema;

    /**
     * The number of predictive features expected by the model.
     */
    private final int numFeatures;

    /**
     * Lock serializing access to the non-thread-safe native booster during prediction.
     */
    private final Object predictLock = new Object();

    /**
     * Constructor.
     *
     * @param booster The trained/loaded native XGBoost booster.
     * @param schema  The {@link DatasetSchema} the model uses.
     */
    XgboostClassificationModel(final Booster booster, final DatasetSchema schema) {
        this.booster = booster;
        this.schema = schema;
        this.numFeatures = XgboostSchemaUtils.numFeatures(schema);
    }

    @Override
    public double[] getClassDistribution(final Instance instance) {
        final float[] row = XgboostSchemaUtils.featureRow(instance, this.schema);

        final float[][] predictions;
        try {
            // The native booster handle is not thread-safe; serialize predictions.
            synchronized (this.predictLock) {
                predictions = this.booster.inplace_predict(row, 1, this.numFeatures, MISSING_VALUE);
            }
        } catch (final XGBoostError e) {
            throw new RuntimeException("XGBoost failed to score the instance.", e);
        }

        return toClassDistribution(predictions[0]);
    }

    @Override
    public int classify(final Instance instance) {
        final double[] distribution = getClassDistribution(instance);

        int argMax = 0;
        for (int i = 1; i < distribution.length; i++) {
            if (distribution[i] > distribution[argMax]) {
                argMax = i;
            }
        }
        return argMax;
    }

    @Override
    public boolean save(final Path dir, final String name) {
        try {
            this.booster.saveModel(dir.resolve(XgboostModelCreator.MODEL_BINARY_RESOURCE_FILE_NAME).toString());
            return true;
        } catch (final XGBoostError e) {
            logger.error("Failed to save XGBoost model {} to {}.", name, dir, e);
            return false;
        }
    }

    @Override
    public DatasetSchema getSchema() {
        return this.schema;
    }

    @Override
    public void close() {
        this.booster.dispose();
    }

    /**
     * Converts a raw XGBoost prediction row into a class distribution aligned with the schema's target
     * classes.
     *
     * <p>For binary objectives XGBoost outputs a single value - the probability of the positive class -
     * which is expanded to {@code [1 - p, p]}. For multi-class objectives ({@code multi:softprob}) the
     * per-class probability vector is returned as-is.
     *
     * @param prediction The raw prediction row for a single instance.
     * @return The class distribution.
     */
    private static double[] toClassDistribution(final float[] prediction) {
        if (prediction.length == 1) {
            final double positiveProbability = prediction[0];
            return new double[]{1.0 - positiveProbability, positiveProbability};
        }

        final double[] distribution = new double[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            distribution[i] = prediction[i];
        }
        return distribution;
    }
}
