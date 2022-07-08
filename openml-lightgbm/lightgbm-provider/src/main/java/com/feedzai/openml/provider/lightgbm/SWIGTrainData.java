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

import com.microsoft.ml.lightgbm.SWIGTYPE_p_float;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_int;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.doubleChunkedArray;
import com.microsoft.ml.lightgbm.floatChunkedArray;
import com.microsoft.ml.lightgbm.int32ChunkedArray;
import com.microsoft.ml.lightgbm.lightgbmlib;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handles train data resources and provides basic operations to manipulate train data.
 *
 *  This class is responsible for initializing, managing and releasing all
 *  Handles train data resources and provides basic operations to manipulate train data.
 *  LightGBM SWIG train resources and resource handlers in a memory-safe manner.
 *
 *  Whatever happens, it guarantees that no memory leaks are left behind.
 *
 *  @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 *  @since 1.0.10
 */
public class SWIGTrainData implements AutoCloseable {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(SWIGTrainData.class);

    /**
     * Handle for the output parameter necessary for the LightGBM dataset instantiation.
     */
    SWIGTYPE_p_p_void swigOutDatasetHandlePtr;

    /**
     * SWIG Features chunked data array.
     * This objects stores elements in float64 format
     * as a list of chunks that it manages automatically.
     *
     * In the current implementation, features are stored in row-major order, i.e.,
     * each instance is stored contiguously.
     */
    doubleChunkedArray swigFeaturesChunkedArray;

    /**
     * SWIG pointer to the labels array (array of float32 elements).
     */
    SWIGTYPE_p_float swigTrainLabelDataArray;

    /**
     * SWIG pointer to the constraint group array (array of float32 elements).
     */
    SWIGTYPE_p_int swigConstraintGroupDataArray;

    /**
     * SWIG LightGBM dataset handle.
     */
    SWIGTYPE_p_void swigDatasetHandle;

    /**
     * Number of features per instance.
     */
    public final int numFeatures;

    /**
     * Number of instances to store in each chunk.
     */
    private final long numInstancesChunk;

    /**
     * SWIG object to hold the labels in chunks.
     */
    floatChunkedArray swigLabelsChunkedArray;

    /**
     * SWIG object to hold the constraint groups in chunks.
     */
    int32ChunkedArray swigConstraintGroupChunkedArray;

    /**
     * Whether the LightGBM model is fairness constrained (aka FairGBM).
     */
    private final boolean fairnessConstrained;

    /**
     * Constructor.
     *
     * Allocates all the initial handles necessary to bootstrap (but not use) the
     * in-memory LightGBM dataset + booster structures.
     *
     * After that the BoosterHandle and the DatasetHandle will still need to be initialized at the proper times:
     * @see SWIGTrainData#initSwigDatasetHandle()
     *
     * @param numFeatures       The number of features.
     * @param numInstancesChunk The number of instances per chunk of data.
     */
    public SWIGTrainData(final int numFeatures, final long numInstancesChunk) {
        this(numFeatures, numInstancesChunk, false);
    }

    /**
     * Constructor.
     *
     * Allocates al the initial ahndles necessary to bootstrap (but not use) the
     * in-memory LightGBM dataset, plus booster structures.
     *
     * If fairnessConstrained=true, this will also include data on which sensitive
     * group each instance belongs to.
     *
     * @param numFeatures           The number of features.
     * @param numInstancesChunk     The number of instances per chunk of data.
     * @param fairnessConstrained   Whether this data will be used for a model with fairness (group) constraints.
     */
    public SWIGTrainData(final int numFeatures, final long numInstancesChunk, final boolean fairnessConstrained) {
        this.numFeatures = numFeatures;
        this.numInstancesChunk = numInstancesChunk;
        this.swigOutDatasetHandlePtr = lightgbmlib.voidpp_handle();
        this.fairnessConstrained = fairnessConstrained;

        logger.debug("Intermediate SWIG train buffers will be allocated in chunks of {} instances.", numInstancesChunk);
        // 1-D Array in row-major-order that stores only the features (excludes label) in double format by chunks:
        this.swigFeaturesChunkedArray = new doubleChunkedArray(numFeatures * numInstancesChunk);

        // 1-D Array with the labels (float32):
        this.swigLabelsChunkedArray = new floatChunkedArray(numInstancesChunk);

        // 1-D Array with the constraint group:
        if (this.fairnessConstrained) {
            this.swigConstraintGroupChunkedArray = new int32ChunkedArray(numInstancesChunk);
        }
    }

    /**
     * Adds a value to the features' ChunkedArray.
     * @param value value to insert.
     */
    public void addFeatureValue(double value) {
        this.swigFeaturesChunkedArray.add(value);
    }

    /**
     * Adds a value to the labels' ChunkedArray.
     */
    public void addLabelValue(float value) {
        this.swigLabelsChunkedArray.add(value);
    }

    /**
     * Adds a value to the constraint group ChunkedArray.
     * @param value the value to add.
     */
    public void addConstraintGroupValue(int value) {
        assert this.fairnessConstrained : "Trying to set constraint group data with fairnessConstrained=false";
        this.swigConstraintGroupChunkedArray.add(value);
    }

    /**
     * @return Return the chunk sizes (in instances).
     */
    public long getNumInstancesChunk() {
        return numInstancesChunk;
    }

    /**
     * Initializes the swigTrainLabelDataArray, copies the chunked label array data to it,
     * and releases the chunked data from memory.
     */
    SWIGTYPE_p_float coalesceChunkedSwigTrainLabelDataArray() {
        this.swigTrainLabelDataArray = lightgbmlib.new_floatArray(this.swigLabelsChunkedArray.get_add_count());
        this.swigLabelsChunkedArray.coalesce_to(this.swigTrainLabelDataArray);
        this.swigLabelsChunkedArray.release();
        return this.swigTrainLabelDataArray;
    }

    /**
     * Initializes the swigConstraintGroupDataArray, copies the chunked constraint group array data to it,
     * and releases the chunked data from memory.
     */
    SWIGTYPE_p_int coalesceChunkedSwigConstraintGroupDataArray() {
        this.swigConstraintGroupDataArray = lightgbmlib.new_intArray(this.swigConstraintGroupChunkedArray.get_add_count());
        this.swigConstraintGroupChunkedArray.coalesce_to(this.swigConstraintGroupDataArray);
        this.swigConstraintGroupChunkedArray.release();
        return this.swigConstraintGroupDataArray;
    }

    /**
     * Setup swigDatasetHandle after its setup.
     */
    void initSwigDatasetHandle() {
        this.swigDatasetHandle = lightgbmlib.voidpp_value(this.swigOutDatasetHandlePtr);
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
     * Release the memory of the constraint group array.
     * This can be called after instantiating the dataset and setting the constraint group in it.
     */
    void destroySwigConstraintGroupDataArray() {

        if (this.swigConstraintGroupDataArray != null) {
            lightgbmlib.delete_intArray(this.swigConstraintGroupDataArray);
            this.swigConstraintGroupDataArray = null;
        }
    }


    /**
     * Release the memory of the chunked features array.
     * This can be called after instantiating the dataset.
     *
     * Although this simply calls `release()`.
     * After this that object becomes unusable.
     * To cleanup and reuse call `clear()` instead.
     */
    void releaseSwigTrainFeaturesChunkedArray() {
        this.swigFeaturesChunkedArray.release();
    }

    /**
     * Releases the ChunkedArrays of both features and label.
     * Idempotent. After calling this ChunkedArrays cannot be re-used.
     */
    void releaseChunkedResources() {
        this.swigFeaturesChunkedArray.release();
        this.swigLabelsChunkedArray.release();

        if (this.swigConstraintGroupChunkedArray != null) {
            this.swigConstraintGroupChunkedArray.release();
        }
    }

    /**
     * Release any allocated resources.
     * This operation is idempotent and can be safely called at any time as many times as you wish.
     */
    @Override
    public void close() {

        releaseChunkedResources();
        destroySwigTrainLabelDataArray();
        destroySwigConstraintGroupDataArray();

        if (this.swigOutDatasetHandlePtr != null) {
            lightgbmlib.delete_voidpp(this.swigOutDatasetHandlePtr);
            this.swigOutDatasetHandlePtr = null;
        }

        if (this.swigDatasetHandle != null) {
            lightgbmlib.LGBM_DatasetFree(this.swigDatasetHandle);
            this.swigDatasetHandle = null;
        }
    }
}
