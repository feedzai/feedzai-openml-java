/*
 * Copyright 2019 Feedzai
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

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import hex.genmodel.GenModel;
import hex.genmodel.easy.prediction.AbstractPrediction;
import hex.genmodel.easy.prediction.AnomalyDetectionPrediction;

import java.io.Closeable;
import java.nio.file.Path;

/**
 * A classification model representation for an Anomaly Detection algorithm.
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 1.0.0
 */
public class AnomalyDetectionClassificationH2OModel extends AbstractClassificationH2OModel {

    /**
     * The score in between the top and bottom score bounds.
     */
    private static final double MID_SCORE = 0.5;

    /**
     * Constructor for a {@link AbstractClassificationH2OModel}.
     *
     * @param genModel  The imported model generated in H2O.
     * @param modelPath The path from where the model was initially loaded.
     * @param schema    The {@link DatasetSchema} the model uses.
     * @param closeable A {@link Closeable} that needs to be closed upon {@link #close()}.
     */
    AnomalyDetectionClassificationH2OModel(final GenModel genModel, final Path modelPath, final DatasetSchema schema, final Closeable closeable) {
        super(genModel, modelPath, schema, closeable);
    }

    @Override
    public double[] getClassDistribution(final Instance instance) {
        final AbstractPrediction abstractPrediction = predictInstance(instance);
        final double score = getCompressedNormalizedScore((AnomalyDetectionPrediction) abstractPrediction);

        // [not anomaly, anomaly]
        return new double[]{1 - score, score};
    }

    /**
     * Converts the score from the {@link AnomalyDetectionPrediction H2O prediction} into the OpenML score.
     *
     * Note that due the fact that H2O's score might ocasionally go over the limits (either to 1 or -1), we apply bound compression,
     * as the OpenML score semantic doesn't allow scores over the limits.
     *
     * @param prediction The H2O Prediction, where the scores are kept.
     * @return A value between [0..1], where 0 represents certain normality and 1 represents certain anomaly.
     * @implNote This method assumes H2O's score to be between -1 and 1 (and occasionally slight out-of-bounds), where -1 represents certain genuineness and 1 represents certain
     * anomaly.
     */
    private double getCompressedNormalizedScore(final AnomalyDetectionPrediction prediction) {
        final double rawScore = prediction.normalizedScore;
        if (rawScore > 1) {
            return 1;
        }
        else if (rawScore < 0) {
            return 0;
        }
        return rawScore;
    }

    @Override
    public int classify(final Instance instance) {
        final AbstractPrediction abstractPrediction = predictInstance(instance);

        final double normalizedScore = getCompressedNormalizedScore((AnomalyDetectionPrediction) abstractPrediction);

        return normalizedScore > MID_SCORE ? 1 : 0;
    }
}
