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
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.microsoft.ml.lightgbm.lightgbmlibJNI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

import static com.feedzai.openml.provider.lightgbm.LightGBMUtils.BINARY_LGBM_NUM_CLASSES;

/**
 * This class is used to wrap any lighgbmlib* calls and expose
 * a simpler interface to LightGBM.
 *
 * <p>This class is <b>ThreadSafe</b>, not allowing for serializing any attempts for parallel scoring.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
class LightGBMSWIG {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(LightGBMSWIG.class);

    /**
     * Stores the index to the target field in the schema.
     */
    private final int schemaTargetIndex;

    /**
     * Number of features in the schema.
     */
    private final int schemaNumFields;

    /**
     * Number of loaded model classes.
     * LightGBM represents binary classification as 1.
     */
    private int boosterNumClasses;

    /**
     * This handles all low-level swig resource handlers
     * creation/destruction to avoid memory issues
     * and guarantee that those operations are correct
     * even when low-level exceptions are thrown.
     */
    private final SWIGResources swigResources;

    /**
     * Will read the model at path and initialize it.
     * If any LightGBM error arises a ModelLoadingException is thrown.
     *
     * @param modelPath          Path to the model
     * @param schema             Input schema
     * @param lightGBMParameters LightGBM parameters
     * @throws ModelLoadingException in case any LightGBM error occurs.
     */
    public LightGBMSWIG(final String modelPath,
                        final DatasetSchema schema, final String lightGBMParameters) throws ModelLoadingException {

        this.schemaNumFields = schema.getFieldSchemas().size();
        this.schemaTargetIndex = schema.getTargetIndex().orElse(-1);
        this.swigResources = new SWIGResources(modelPath, lightGBMParameters);

        initBoosterNumClasses();
    }


    /**
     * From the input instance, copies the values into the
     * SWIG Instance Array so it can be scored by LightGBM.
     *
     * <p>Skips the label (if it exists in the instance) and copies only the features.
     *
     * <p><b>Note</b> that this method is not thread safe (by itself) and thus needs to be called in a synchronized
     * manner.
     *
     * @param instance The instance from pulse.
     */
    private void copyDataToSWIGInstance(final Instance instance) {

        int skipTargetOffset = 0; // set to 1 after passing the target (if it exists)
        for (int i = 0; i < this.schemaNumFields; ++i) {
            // If the label is not present, targetIndex=-1, thus "if" won't trigger:
            if (i == this.schemaTargetIndex) {
                skipTargetOffset = -1;
            } else {
                lightgbmlibJNI.doubleArray_setitem(
                        this.swigResources.swigInstancePtr,
                        i + skipTargetOffset,
                        instance.getValue(i)
                );
            }
        }
    }

    /**
     * Returns the class distribution scores for the current instance.
     *
     * @param instance The instance from pulse.
     * @return double[2] array with scores.
     */
    public double[] getBinaryClassDistribution(final Instance instance) {

        // we need to lock the resources to avoid having multiple threads sharing the same swig resources.
        synchronized (this.swigResources) {

            copyDataToSWIGInstance(instance);
            final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterPredictForMatSingleRowFast(
                    this.swigResources.swigFastConfigHandle,
                    this.swigResources.swigInstancePtr,
                    this.swigResources.swigOutLengthInt64Ptr,
                    this.swigResources.swigOutScoresPtr // preallocated memory
            );

            if (returnCodeLGBM == -1)
                throw new LightGBMException();

            final double predictionScore = lightgbmlibJNI.doubleArray_getitem(this.swigResources.swigOutScoresPtr, 0);
            logger.trace("Prediction: {}", predictionScore);
            return new double[]{1 - predictionScore, predictionScore};
        }
    }



    /**
     * Gets number of features in the model.
     *
     * @return Number of features in the model (retrieved from model binary).
     */
    public int getBoosterNumFeatures() {
        return this.swigResources.getBoosterNumFeatures();
    }

    /**
     * Gets the name of the features in the model.
     *
     * @return Names of the features in the model (retrieved from model binary).
     * @since 1.0.18
     */
    public String[] getBoosterFeatureNames() {
        return this.swigResources.getBoosterFeatureNames();
    }

    /**
     * Initializes the number of model classes from the model binary.
     *
     * <p><b>Note</b> that this method is not thread safe (by itself) and thus needs to be called in a synchronized
     * manner.
     *
     * @throws LightGBMException in case there's an error in the C++ core library.
     */
    private void initBoosterNumClasses() throws LightGBMException {
        final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterGetNumClasses(this.swigResources.swigBoosterHandle,
                this.swigResources.swigOutIntPtr);
        if (returnCodeLGBM == -1)
            throw new LightGBMException();
        this.boosterNumClasses = lightgbmlibJNI.intp_value(this.swigResources.swigOutIntPtr);
    }

    /**
     * Gets the number of model classes.
     *
     * @return Number of model classes (retrieved from model binary).
     *         NOTE: It's 1 for binary case in LightGBM!
     */
    public int getBoosterNumClasses() {
        return this.boosterNumClasses;
    }

    /**
     * Checks if the model is binary.
     *
     * @return Returns true if the model is binary.
     */
    public boolean isModelBinary() {
        return this.boosterNumClasses == BINARY_LGBM_NUM_CLASSES;
    }

    /**
     * Gets the number of booster iterations in the model.
     *
     * @return The number of booster iterations in the model (retrieved from model binary).
     */
    public int getBoosterNumIterations() { return this.swigResources.getBoosterNumIterations(); }

    /**
     * Saves the model to disk.
     *
     * @param outputModelFilePath the path of the output LightGBM model file.
     */
    public void saveModelToDisk(final Path outputModelFilePath) {

        logger.info("Saving model to disk.");
        logger.debug("Saving model to disk @ {}.", outputModelFilePath);

        final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterSaveModel(
                this.swigResources.swigBoosterHandle,
                0,
                -1,
                lightgbmlibJNI.C_API_FEATURE_IMPORTANCE_GAIN_get(),
                outputModelFilePath.toAbsolutePath().toString()
        );
        if (returnCodeLGBM == -1) {
            logger.error("Could not save model to disk.");
            throw new LightGBMException();
        }
    }
}
