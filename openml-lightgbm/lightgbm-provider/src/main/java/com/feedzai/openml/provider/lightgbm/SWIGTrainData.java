package com.feedzai.openml.provider.lightgbm;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_double;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_float;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handles train data resources and provides basic operations to manipulate train data.
 */
public class SWIGTrainData implements AutoCloseable {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(SWIGTrainData.class);

    /**
     * Handle for the output parameter necessary for the LightGBM dataset instantiation.
     */
    public SWIGTYPE_p_p_void swigOutDatasetHandlePtr;

    /**
     * SWIG pointer to the Features data array.
     * This array stores elements in float64 format.
     *
     * In the current implementation, features are stored in row-major order, i.e.,
     * each instance is stored contiguously.
     */
    public SWIGTYPE_p_double swigTrainFeaturesDataArray;

    /**
     * SWIG pointer to the labels array (array of float32 elements).
     */
    public SWIGTYPE_p_float swigTrainLabelDataArray;

    /**
     * SWIG LightGBM dataset handle.
     */
    public SWIGTYPE_p_void swigDatasetHandle;

    private final int chunkSize;

    /**
     * Constructor.
     *
     * Allocates all the initial handles necessary to bootstrap (but not use) the
     * in-memory LightGBM dataset + booster structures.
     *
     * After that the BoosterHandle and the DatasetHandle will still need to be initialized at the proper times:
     * @see SWIGTrainData#initSwigDatasetHandle()
     *
     * @param numFeatures   The number of features.
     */
    public SWIGTrainData(final int numFeatures, final int chunkSize) {
        this.chunkSize = chunkSize;
        this.swigOutDatasetHandlePtr = lightgbmlib.voidpp_handle();

        logger.debug("Allocating SWIG train data array.");
        // 1-D Array in row-major-order that stores only the features (excludes label) in double format:
        /* TODO: FTL
        this.swigTrainFeaturesDataArray = lightgbmlib.new_doubleArray(((long)numInstances) * ((long)numFeatures));
        // 1-D Array with the labels (float32):
        this.swigTrainLabelDataArray = lightgbmlib.new_floatArray(numInstances);
        */
    }

    /**
     * Setup swigDatasetHandle after its setup.
     */
    public void initSwigDatasetHandle() {
        this.swigDatasetHandle = lightgbmlib.voidpp_value(this.swigOutDatasetHandlePtr);
    }

    /**
     * Release the memory of the label array.
     * This can be called after instantiating the dataset and setting the label in it.
     */
    public void destroySwigTrainLabelDataArray() {

        if (this.swigTrainLabelDataArray != null) {
            lightgbmlib.delete_floatArray(this.swigTrainLabelDataArray);
            this.swigTrainLabelDataArray = null;
        }
    }

    /**
     * Release the memory of the features array.
     * This can be called after instantiating the dataset.
     */
    public void destroySwigTrainFeaturesDataArray() {

        if (this.swigTrainFeaturesDataArray != null) {
            lightgbmlib.delete_doubleArray(this.swigTrainFeaturesDataArray);
            this.swigTrainFeaturesDataArray = null;
        }
    }

    /**
     * Release any allocated resources.
     * This operation is idempotent and can be safely called at any time as many times as you wish.
     */
    public void releaseResources() {

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
    }

    @Override
    public void close() throws Exception {
        releaseResources();
    }
}
