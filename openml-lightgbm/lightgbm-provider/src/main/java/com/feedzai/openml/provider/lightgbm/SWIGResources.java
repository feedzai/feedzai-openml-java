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

import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;
import com.microsoft.ml.lightgbm.lightgbmlibJNI;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static com.feedzai.openml.provider.lightgbm.LightGBMUtils.BINARY_LGBM_NUM_CLASSES;

/**
 * This class is responsible for initializing, managing and releasing all
 * LightGBM SWIG resources and resource handlers in a memory-safe manner.
 * <p>
 * Whatever happens, it guarantees that no memory leaks are left behind.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
class SWIGResources implements AutoCloseable {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(SWIGResources.class);

    /**
     * SWIG pointer to BoosterHandle.
     */
    public Long swigBoosterHandle;

    /**
     * SWIG pointer to FastConfigHandle.
     */
    public Long swigFastConfigHandle;

    /**
     * SWIG pointer to FastConfigContributionsHandle.
     *
     * @since 1.2.2
     */
    public Long swigFastConfigContributionsHandle;

    /**
     * Useless variable in the C API, for we already have to preallocate swigOutScoresPtr,
     * so we get no information from this API output.
     */
    public Long swigOutLengthInt64Ptr;

    /**
     * Pointer to the instance array.
     */
    public Long swigInstancePtr;

    /**
     * SWIG Pointer to the scored array output (pre-allocated by us).
     */
    public Long swigOutScoresPtr;

    /**
     * SWIG Pointer to the features contributions array output (pre-allocated by us).
     *
     * @since 1.2.2
     */
    public Long swigOutContributionsPtr;

    /**
     * A handler to collect a pointer to int output.
     * Reusable handler for any swig OutIntPtr call.
     * As many calls output an IntPtr, this field is useful
     * to avoid repeating pointer allocation/destruction ops.
     */
    public Long swigOutIntPtr;

    /**
     * Number of iterations in the trained LightGBM boosting model.
     * Whilst not a swig resource, it is automatically retrieved during model loading,
     * thus we store it to avoid calling it again.
     */
    private Integer boosterNumIterations;

    /**
     * Number of features in the trained LightGBM boosting model.
     * Whilst not a swig resource, it is automatically retrieved during model loading,
     * thus we store it to avoid calls that can fail.
     */
    private Integer boosterNumFeatures;

    /**
     * Names of features in the trained LightGBM boosting model.
     * Whilst not a swig resource, it is automatically retrieved during model loading,
     * thus we store it to avoid calls that can fail.
     *
     * @since 1.0.18
     */
    private String[] boosterFeatureNames;

    /**
     * Constructor. Initializes a model handle and all resource handlers
     * necessary to implement the standard model operations.
     *
     * @param modelPath          Path to the model folder.
     * @param lightGBMParameters String with LightGBM parameters.
     * @throws ModelLoadingException Error loading the model.
     * @throws LightGBMException     in case there's an error in the C++ core library.
     */
    public SWIGResources(final String modelPath,
                         final String lightGBMParameters) throws ModelLoadingException, LightGBMException {

        this.swigOutIntPtr = lightgbmlibJNI.new_intp();
        initModelResourcesFromFile(modelPath);
        initAuxiliaryModelResources();
        initBoosterFastPredictHandle(lightGBMParameters);
        initBoosterFastContributionsHandle(lightGBMParameters);
    }

    /**
     * Releases the SWIG resources and throws a ModelLoadingException with a message prefix.
     *
     * @param msgPrefix Launched exception's error message prefix.
     * @throws ModelLoadingException Exception thrown when this method is called.
     */
    private void releaseResourcesAndThrowModelLoadingException(final String msgPrefix) throws ModelLoadingException {

        releaseInitializedSWIGResources();
        throw new ModelLoadingException(msgPrefix + lightgbmlibJNI.LGBM_GetLastError());
    }

    /**
     * Loads the LightGBM model from the file.
     * Initializes some variables like the number of iterations
     * (trees) in the boosting model
     * and returns the boosterNumIterations which is 0
     * if anything failed (besides the thrown ModelLoadingException).
     *
     * @param modelPath Filepath of the LightGBM model in disk.
     *
     * @throws ModelLoadingException in case LGBM_BoosterCreateFromModelfile fails.
     */
    private void initModelResourcesFromFile(final String modelPath) throws ModelLoadingException {

        final long swigOutBoosterHandlePtr = lightgbmlibJNI.voidpp_handle();

        try {
            final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterCreateFromModelfile(
                    modelPath,
                    this.swigOutIntPtr,
                    swigOutBoosterHandlePtr
            );
            if (returnCodeLGBM == -1) {
                releaseResourcesAndThrowModelLoadingException("Error loading LightGBM model from file: ");
            }
            logger.debug("Loaded LightGBM model from file.");

            this.swigBoosterHandle = lightgbmlibJNI.voidpp_value(swigOutBoosterHandlePtr);
            this.boosterNumIterations = lightgbmlibJNI.intp_value(this.swigOutIntPtr);
        } finally {
            lightgbmlibJNI.delete_voidpp(swigOutBoosterHandlePtr);
        }
    }

    /**
     * Initializes the FastConfig object from a Booster.
     * This FastConfig is responsible for caching Booster+Prediction settings for repeated prediction calls.
     * It is used in the *Fast() predict methods instead of the Booster and prediction settings.
     *
     * @param LightGBMParameters String with custom LightGBM parameters.
     *
     * @throws ModelLoadingException in case there is a C++ back-end error creating the FastConfig object.
     */
    private void initBoosterFastPredictHandle(final String LightGBMParameters) throws ModelLoadingException {

        final long swigOutFastConfigHandlePtr = lightgbmlibJNI.voidpp_handle();

        try {
            final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterPredictForMatSingleRowFastInit(
                    swigBoosterHandle,
                    lightgbmlibConstants.C_API_PREDICT_NORMAL,
                    0, // startIteration = 0 to use all iterations.
                    -1, // numIterations = all.
                    lightgbmlibConstants.C_API_DTYPE_FLOAT64,
                    getBoosterNumFeatures(),
                    LightGBMParameters,
                    swigOutFastConfigHandlePtr
            );

            if (returnCodeLGBM == -1) {
                releaseResourcesAndThrowModelLoadingException("Error initializing prediction FastConfig settings: ");
            }

            this.swigFastConfigHandle = lightgbmlibJNI.voidpp_value(swigOutFastConfigHandlePtr);
        } finally {
            lightgbmlibJNI.delete_voidpp(swigOutFastConfigHandlePtr);
        }
    }

    /**
     * Initializes the FastConfigContributions object from a Booster.
     * This FastConfig is responsible for caching Booster+Contributions settings for repeated prediction calls.
     * It is used in the *Fast() predict methods instead of the Booster and prediction settings.
     *
     * @param LightGBMParameters String with custom LightGBM parameters.
     * @since 1.2.2
     * @throws ModelLoadingException in case there is a C++ back-end error creating the FastConfigContributions object.
     */
    private void initBoosterFastContributionsHandle(final String LightGBMParameters) throws ModelLoadingException {

        final long swigOutFastConfigHandlePtr = lightgbmlibJNI.voidpp_handle();

        try {
            final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterPredictForMatSingleRowFastInit(
                    swigBoosterHandle,
                    lightgbmlibConstants.C_API_PREDICT_CONTRIB,
                    0, // startIteration = 0 to use all iterations.
                    -1, // numIterations = all.
                    lightgbmlibConstants.C_API_DTYPE_FLOAT64,
                    getBoosterNumFeatures(),
                    LightGBMParameters,
                    swigOutFastConfigHandlePtr
            );

            if (returnCodeLGBM == -1) {
                releaseResourcesAndThrowModelLoadingException("Error initializing prediction FastConfig settings: ");
            }

            this.swigFastConfigContributionsHandle = lightgbmlibJNI.voidpp_value(swigOutFastConfigHandlePtr);
        } finally {
            lightgbmlibJNI.delete_voidpp(swigOutFastConfigHandlePtr);
        }
    }

    /**
     * Assumes the model was already loaded from file.
     * Initializes the remaining SWIG resources needed to use the model.
     *
     * @throws LightGBMException in case there's an error in the C++ core library.
     */
    private void initAuxiliaryModelResources() throws LightGBMException {

        this.boosterNumFeatures = computeBoosterNumFeaturesFromModel();
        logger.debug("Loaded LightGBM Model has {} features.", this.boosterNumFeatures);

        this.boosterFeatureNames = computeBoosterFeatureNamesFromModel();

        this.swigOutLengthInt64Ptr = lightgbmlibJNI.new_int64_tp();
        this.swigInstancePtr = lightgbmlibJNI.new_doubleArray(getBoosterNumFeatures());
        this.swigOutScoresPtr = lightgbmlibJNI.new_doubleArray(BINARY_LGBM_NUM_CLASSES);
        this.swigOutContributionsPtr = lightgbmlibJNI.new_doubleArray(this.boosterNumFeatures);
    }

    /**
     * If partial resource initialization was performed in
     * the constructor and any allocated handlers should be freed,
     * you can call this function to release them.
     * <p>
     * You can call this function as many times as you want,
     * it will safely execute without double-releasing resources.
     * <p>
     * Note: This will be called in close() automatically at the right time
     * if the constructor finished properly at the time of GC.
     * <p>
     *
     * @throws LightGBMException in case there's any lightGBM error.
     */
    private void releaseInitializedSWIGResources() throws LightGBMException {

        logger.trace("Releasing SWIG resources!");

        // Delete resource handlers:
        if (this.swigOutLengthInt64Ptr != null) {
            lightgbmlibJNI.delete_int64_tp(this.swigOutLengthInt64Ptr);
            this.swigOutLengthInt64Ptr = null;
        }
        if (this.swigInstancePtr != null) {
            lightgbmlibJNI.delete_doubleArray(this.swigInstancePtr);
            this.swigInstancePtr = null;
        }
        if (this.swigOutScoresPtr != null) {
            lightgbmlibJNI.delete_doubleArray(this.swigOutScoresPtr);
            this.swigOutScoresPtr = null;
        }
        if (this.swigOutIntPtr != null) {
            lightgbmlibJNI.delete_intp(this.swigOutIntPtr);
            this.swigOutIntPtr = null;
        }
        if (this.swigOutContributionsPtr != null) {
            lightgbmlibJNI.delete_doubleArray(this.swigOutContributionsPtr);
            this.swigOutContributionsPtr = null;
        }

        // Delete FastConfig configuration resource:
        if (this.swigFastConfigHandle != null) {
            final int returnCodeLGBM = lightgbmlibJNI.LGBM_FastConfigFree(this.swigFastConfigHandle);
            this.swigFastConfigHandle = null;

            if (returnCodeLGBM == -1) {
                throw new LightGBMException();
            }
        }
        if (this.swigFastConfigContributionsHandle != null) {
            final int returnCodeLGBM = lightgbmlibJNI.LGBM_FastConfigFree(this.swigFastConfigContributionsHandle);
            this.swigFastConfigContributionsHandle = null;

            if (returnCodeLGBM == -1) {
                throw new LightGBMException();
            }
        }

        // Delete model resources:
        if (this.swigBoosterHandle != null) {
            final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterFree(this.swigBoosterHandle);
            this.swigBoosterHandle = null;

            if (returnCodeLGBM == -1) {
                throw new LightGBMException();
            }
        }
    }

    /**
     * Automatically called at the end.
     * Will release any initialized resource handlers.
     */
    @Override
    public void close() {
        releaseInitializedSWIGResources();
    }

    /**
     * Returns the number of iterations in the LightGBM model.
     *
     * @return boosterNumIterations
     */
    public int getBoosterNumIterations() { return this.boosterNumIterations; }

    /**
     * Returns the number of features in the LightGBM model.
     *
     * @return boosterNumIterations
     */
    public int getBoosterNumFeatures() { return this.boosterNumFeatures; }

    /**
     * Returns the name of features in the LightGBM model.
     *
     * @return boosterFeatureNames
     * @since 1.0.18
     */
    public String[] getBoosterFeatureNames() {
        return this.boosterFeatureNames;
    }

    /**
     * Computes the number of features in the model and returns it.
     *
     * @throws LightGBMException when there is a LightGBM C++ error.
     * @returns int with the number of Booster features.
     */
    private Integer computeBoosterNumFeaturesFromModel() throws LightGBMException {

        final int returnCodeLGBM = lightgbmlibJNI.LGBM_BoosterGetNumFeature(
                this.swigBoosterHandle,
                this.swigOutIntPtr);
        if (returnCodeLGBM == -1)
            throw new LightGBMException();

        return lightgbmlibJNI.intp_value(this.swigOutIntPtr);
    }

    /**
     * Compute the feature names, from the model.
     *
     * @throws LightGBMException when there is a LightGBM C++ error.
     * @return a string array with the feature names.
     * @since 1.0.18
     */
    private String[] computeBoosterFeatureNamesFromModel() throws LightGBMException {

        final long swigStringArrayHandle = lightgbmlibJNI.LGBM_BoosterGetFeatureNamesSWIG(this.swigBoosterHandle);

        if (swigStringArrayHandle == 0) {
            logger.error("Could not read feature names.");
            throw new LightGBMException();
        }

        final String[] featureNames = lightgbmlibJNI.StringArrayHandle_get_strings(swigStringArrayHandle);
        logger.debug("LightGBM model features: {}.", Arrays.toString(featureNames));

        logger.trace("Deallocating feature names.");
        lightgbmlibJNI.StringArrayHandle_free(swigStringArrayHandle);

        return featureNames;
    }
}
