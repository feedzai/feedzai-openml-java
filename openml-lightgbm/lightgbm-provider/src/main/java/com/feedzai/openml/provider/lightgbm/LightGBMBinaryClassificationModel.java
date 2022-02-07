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

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.model.ClassificationMLModel;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/**
 * This class is responsible for loading a saved LightGBM model binary and scoring instances.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
public class LightGBMBinaryClassificationModel implements ClassificationMLModel {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(LightGBMBinaryClassificationModel.class);

    /**
     * The schema used to load this model.
     */
    private final DatasetSchema schema;

    /**
     * This takes care of all calls to the LightGBM SWIG library
     * and takes care of handling resources allocation/destruction.
     */
    private final LightGBMSWIG lgbm;

    /**
     * String with LightGBM parameters used for prediction.
     */
    private static final String LIGHTGBM_PREDICTION_PARAMETERS = "num_threads=1";

    /**
     * Constructor.
     * Takes care of loading the model from the file and initializing all SWIG-LGBM resources for prediction.
     *
     * @param modelPath Path to the LightGBM model binary.
     * @param schema    The schema of the loaded model.
     * @throws ModelLoadingException if there was an error loading the model resources in the LightGBM low-level code.
     */
    LightGBMBinaryClassificationModel(final Path modelPath, final DatasetSchema schema) throws ModelLoadingException {
        this.schema = schema;
        lgbm = new LightGBMSWIG(modelPath.toString(), schema, LIGHTGBM_PREDICTION_PARAMETERS);
    }

    @Override
    public double[] getClassDistribution(final Instance instance) { return lgbm.getBinaryClassDistribution(instance); }

    @Override
    public int classify(final Instance instance) {
        // Assuming binary classification:
        double score0 = getClassDistribution(instance)[0];
        if (score0 > 0.5)
            return 0;
        else
            return 1;
    }

    @Override
    public boolean save(final Path dir, final String name) {

        try {
            lgbm.saveModelToDisk(dir.resolve(LightGBMModelCreator.MODEL_BINARY_RESOURCE_FILE_NAME));
        } catch (final Exception e) {
            logger.error("Failed to save model to disk: {}", e.getMessage());
            return false;
        }
        return true;
    }

    @Override
    public DatasetSchema getSchema() {
        return this.schema;
    }

    @Override
    public void close() throws Exception {
    }

    /**
     * Get features contributions double [ ].
     *
     * @param instance the instance
     * @return the double [ ]
     * @deprecated use {@link LightGBMExplanationsAlgorithm#getFeatureContributions(Instance)} instead.
     */
    public double[] getFeatureContributions(final Instance instance) {
        return this.lgbm.getFeaturesContributions(instance);
    }

    /**
     * @return Number of features used in the model.
     */
    public int getBoosterNumFeatures() {
        return lgbm.getBoosterNumFeatures();
    }

    /**
     * @return Names of the features in the model.
     * @since 1.0.18
     */
    public String[] getBoosterFeatureNames() {
        return lgbm.getBoosterFeatureNames();
    }

    /**
     * @return The number of booster iterations in the model.
     */
    public int getBoosterNumIterations() { return lgbm.getBoosterNumIterations(); }

    /**
     * @return Returns true if the model is binary.
     */
    public boolean isModelBinary() {
        return lgbm.isModelBinary();
    }

}
