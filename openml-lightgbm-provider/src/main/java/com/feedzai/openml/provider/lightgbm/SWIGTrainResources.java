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

import com.microsoft.ml.lightgbm.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class is responsible for initializing, managing and releasing all
 * LightGBM SWIG train resources and resource handlers in a memory-safe manner.
 *
 * Whatever happens, it guarantees that no memory leaks are left behind.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 0.8.0
 */
public class SWIGTrainResources implements  AutoCloseable {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(SWIGTrainResources.class);

    /**
     * Handle for the output parameter necessary for the LightGBM dataset instantiation.
     */
    SWIGTYPE_p_p_void swigOutDatasetHandlePtr;

    /**
     * SWIG pointer to the Features data array.
     * This array stores elements in float64 format.
     *
     * In the current implementation, features are stored in row-major order, i.e.,
     * each instance is stored contiguously.
     */
    SWIGTYPE_p_double swigTrainFeaturesDataArray;

    /**
     * SWIG pointer to the labels array (array of float32 elements).
     */
    SWIGTYPE_p_float swigTrainLabelDataArray;

    /**
     * SWIG LightGBM dataset handle.
     */
    SWIGTYPE_p_void swigDatasetHandle;

    /**
     * SWIG pointer to the output LightGBM Booster Handle during Booster structure instantiation.
     */
    SWIGTYPE_p_p_void swigOutBoosterHandlePtr;

    /**
     * Handle of the LightGBM boosting model post-instantiation.
     */
    SWIGTYPE_p_void swigBoosterHandle;

    /**
     * Constructor.
     *
     * Allocates all the initial handles necessary to bootstrap (but not use) the
     * in-memory LightGBM dataset + booster structures.
     *
     * After that the BoosterHandle and the DatasetHandle will still need to be initialized at the proper times:
     * @see SWIGTrainResources#initSwigDatasetHandle()
     * @see SWIGTrainResources#initSwigBoosterHandle()
     *
     * @param numInstances  The number of instances.
     * @param numFeatures   The number of features.
     */
    public SWIGTrainResources(final int numInstances, final int numFeatures) {
        this.swigOutDatasetHandlePtr = lightgbmlib.voidpp_handle();

        logger.debug("Allocating SWIG train data array.");
        // TODO: https://issues.feedzai.com/browse/PULSEDEV-30932 use memory-allocation exception-safe alternatives
        // 1-D Array in row-major-order that stores only the features (excludes label) in double format:
        this.swigTrainFeaturesDataArray = lightgbmlib.new_doubleArray(numInstances * numFeatures);
        // 1-D Array with the labels (float32):
        this.swigTrainLabelDataArray = lightgbmlib.new_floatArray(numInstances);

        this.swigOutBoosterHandlePtr = lightgbmlib.voidpp_handle();
    }

    /**
     * Setup swigDatasetHandle after its setup.
     */
    void initSwigDatasetHandle() {
        this.swigDatasetHandle = lightgbmlib.voidpp_value(this.swigOutDatasetHandlePtr);
    }

    /**
     * Setup swigBoosterHandle after its structure was created in-memory.
     */
    void initSwigBoosterHandle() {
        this.swigBoosterHandle = lightgbmlib.voidpp_value(this.swigOutBoosterHandlePtr);
    }

    /**
     * Release the memory of the label array.
     * This can be called after instantiating the dataset and setting the label in it.
     */
    void destroySwigTrainLabelDataArray() {

        if (this.swigTrainLabelDataArray != null) {
            lightgbmlib.delete_floatArray(this.swigTrainLabelDataArray);
            this.swigTrainLabelDataArray = null;
        }
    }

    /**
     * Release the memory of the features array.
     * This can be called after instantiating the dataset.
     */
    void destroySwigTrainFeaturesDataArray() {

        if (this.swigTrainFeaturesDataArray != null) {
            lightgbmlib.delete_doubleArray(this.swigTrainFeaturesDataArray);
            this.swigTrainFeaturesDataArray = null;
        }
    }

    /**
     * Release any allocated resources.
     * This operation is idempotent and can be safely called at any time as many times as you wish.
     */
    void releaseResources() {

        if (this.swigOutDatasetHandlePtr != null) {
            lightgbmlib.delete_voidpp(this.swigOutDatasetHandlePtr);
            this.swigOutDatasetHandlePtr = null;
        }

        destroySwigTrainFeaturesDataArray();
        destroySwigTrainLabelDataArray();

        if (this.swigDatasetHandle != null) {
            lightgbmlib.LGBM_DatasetFree(this.swigDatasetHandle);
            this.swigDatasetHandle = null;
        }

        if (this.swigOutBoosterHandlePtr != null) {
            lightgbmlib.delete_voidpp(this.swigOutBoosterHandlePtr);
            this.swigOutBoosterHandlePtr = null;
        }

        if (this.swigBoosterHandle != null) {
            lightgbmlib.LGBM_BoosterFree(this.swigBoosterHandle);
            this.swigBoosterHandle = null;
        }

    }

    @Override
    public void close() throws Exception {
        releaseResources();
    }
}
