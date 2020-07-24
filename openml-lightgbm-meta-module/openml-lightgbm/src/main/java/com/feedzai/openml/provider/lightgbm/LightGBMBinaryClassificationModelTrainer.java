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

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterators;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_double;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_float;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_int;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.NUM_ITERATIONS_PARAMETER_NAME;
import static java.lang.Integer.parseInt;
import static java.util.stream.Collectors.toList;

/**
 * Class to train a LightGBM model.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
final class LightGBMBinaryClassificationModelTrainer {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(LightGBMBinaryClassificationModelTrainer.class);


    /**
     * This class is not meant to be instantiated.
     */
    private LightGBMBinaryClassificationModelTrainer() {}

    /**
     * Train a LightGBM model from scratch, using train data from an in-memory dataset.
     *
     * @param dataset Train dataset
     * @param params LightGBM model parameters
     * @param outputModelFilePath Output filepath for the model file in .txt format.
     */
    static void fit(final Dataset dataset, final Map<String, String> params, final Path outputModelFilePath) {

        final DatasetSchema schema = dataset.getSchema();
        final int numFeatures = schema.getPredictiveFields().size();
        // This shouldn't be needed, but at this level of abstraction, Dataset does not provide a method for size:
        final int numInstances = Iterators.size(dataset.getInstances());
        logger.debug("The train dataset has {} instances.", numInstances);
        final List<Integer> categoricalFeatureIndicesWithoutLabel = getCategoricalFeaturesIndicesWithoutLabel(schema);
        final String trainParams = getLightGBMTrainParamsString(params, categoricalFeatureIndicesWithoutLabel);
        final int numIterations = parseInt(params.get(NUM_ITERATIONS_PARAMETER_NAME));
        logger.debug("LightGBM model trainParams: {}", trainParams);

        final SWIGTrainResources swigTrainResources = new SWIGTrainResources(numInstances, numFeatures);

        /// Create LightGBM dataset
        createTrainDataset(dataset, numFeatures, numInstances, trainParams, swigTrainResources);

        /// Create Booster from dataset
        createBoosterStructure(swigTrainResources, trainParams);
        trainBooster(swigTrainResources.swigBoosterHandle, numIterations);

        /// Save model
        saveModelFileToDisk(swigTrainResources.swigBoosterHandle, outputModelFilePath);
        swigTrainResources.releaseResources(); // Explicitly release C++ resources right away as they're no longer needed.
    }

    /**
     * @param schema Schema of the dataset.
     * @return Returns the indices of the categorical features as if the label didn't exist in the schema.
     */
    private static List<Integer> getCategoricalFeaturesIndicesWithoutLabel(final DatasetSchema schema) {

        final List<FieldSchema> featureFields = schema.getPredictiveFields();
        final int targetIndex = schema.getTargetIndex().get(); // Our model is supervised, and needs the target.

        return featureFields.stream()
                .filter(field -> field.getValueSchema() instanceof CategoricalValueSchema)
                .map(field -> {
                    int fieldIndex = field.getFieldIndex();
                    int fieldAfterLabelOffset = fieldIndex > targetIndex ? -1 : 0;
                    return fieldIndex + fieldAfterLabelOffset; // exclude the label from the indexing.
                })
                .collect(toList());
    }

    /**
     * @param fields List of FieldSchema fields.
     * @return Names of the fields in the input list.
     */
    private static String[] getFieldNames(final List<FieldSchema> fields) {
        return fields.stream().map(FieldSchema::getFieldName).toArray(String[]::new);
    }

    /**
     * Creates a LightGBM training dataset from scratch, based on features+label arrays and train params.
     * <p>
     * This includes:
     * - Creating the dataset structure (with settings and train params);
     * - Initializing the dataset features data + releasing the features memory array;
     * - Initializing the label data in the dataset + releasing the label array;
     * - Setting the feature names in the dataset.
     *
     * @param dataset            Dataset
     * @param numFeatures        Number of features
     * @param numInstances       Number of instances in dataset
     * @param trainParams        LightGBM-formatted params string ("key1=value1 key2=value2 ...")
     * @param swigTrainResources SWIGTrainResources object
     */
    private static void createTrainDataset(final Dataset dataset,
                                           final int numFeatures,
                                           final int numInstances,
                                           final String trainParams,
                                           final SWIGTrainResources swigTrainResources) {

        logger.info("Creating LightGBM dataset");

        logger.debug("Copying train data through SWIG.");
        copyTrainDataToSWIGArrays(
                dataset,
                swigTrainResources.swigTrainFeaturesDataArray,
                swigTrainResources.swigTrainLabelDataArray
        );

        initializeLightGBMTrainDataset(
                swigTrainResources.swigOutDatasetHandlePtr,
                numFeatures,
                numInstances,
                trainParams,
                swigTrainResources.swigTrainFeaturesDataArray
        );
        swigTrainResources.initSwigDatasetHandle();
        swigTrainResources.destroySwigTrainFeaturesDataArray();

        setLightGBMDatasetLabelData(
                swigTrainResources.swigDatasetHandle,
                swigTrainResources.swigTrainLabelDataArray,
                numInstances
        );
        swigTrainResources.destroySwigTrainLabelDataArray();

        setLightGBMDatasetFeatureNames(swigTrainResources.swigDatasetHandle, dataset.getSchema());

        logger.info("Created LightGBM dataset.");
    }

    /**
     * Initializes the LightGBM dataset structure and copies the feature data.
     *
     * @param swigOutDatasetHandlePtr    Generated SWIG output dataset handle pointer.
     * @param numFeatures                Number of features used to predict.
     * @param numInstances               Number of instances in the dataset.
     * @param trainParams                LightGBM string with the train params ("key1=value1 key2=value2 ...").
     * @param swigTrainFeaturesDataArray SWIG pointer to the features array (in row-major order).
     */
    private static void initializeLightGBMTrainDataset(final SWIGTYPE_p_p_void swigOutDatasetHandlePtr,
                                                       final int numFeatures,
                                                       final int numInstances,
                                                       final String trainParams,
                                                       final SWIGTYPE_p_double swigTrainFeaturesDataArray) {

        logger.debug("Initializing LightGBM in-memory structure and setting feature data.");
        final int rowMajor = 1; // Feature data was copied in row-major format.
        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetCreateFromMat(
                lightgbmlib.double_to_voidp_ptr(swigTrainFeaturesDataArray), // Feature data is in row-major format.
                lightgbmlibConstants.C_API_DTYPE_FLOAT64,
                numInstances,
                numFeatures,
                rowMajor,
                trainParams,
                null, // No dataset for alignment with, use a null pointer.
                swigOutDatasetHandlePtr
        );
        if (returnCodeLGBM == -1) {
            logger.error("Could not create LightGBM dataset.");
            throw new LightGBMException();
        }
    }

    /**
     * Sets the LightGBM dataset label data.
     *
     * @param swigDatasetHandle       SWIG Dataset Handle
     * @param swigTrainLabelDataArray SWIG labels array pointer
     * @param numInstances            Number of instances in dataset.
     */
    private static void setLightGBMDatasetLabelData(final SWIGTYPE_p_void swigDatasetHandle,
                                            final SWIGTYPE_p_float swigTrainLabelDataArray,
                                            final int numInstances) {

        logger.debug("Setting label data.");
        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetSetField(
                swigDatasetHandle,
                "label", // LightGBM label column type.
                lightgbmlib.float_to_voidp_ptr(swigTrainLabelDataArray),
                numInstances,
                lightgbmlibConstants.C_API_DTYPE_FLOAT32
        );
        if (returnCodeLGBM == -1) {
            logger.error("Could not set label.");
            throw new LightGBMException();
        }
    }

    /**
     * Sets the feature names on the LightGBM dataset structure.
     *
     * @param swigDatasetHandle SWIG dataset handle
     * @param schema            Dataset schema
     */
    private static void setLightGBMDatasetFeatureNames(final SWIGTYPE_p_void swigDatasetHandle, final DatasetSchema schema) {

        final int numFeatures = schema.getPredictiveFields().size();

        final String[] featureNames = getFieldNames(schema.getPredictiveFields());
        logger.debug("featureNames {}", Arrays.toString(featureNames));

        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetSetFeatureNames(swigDatasetHandle, featureNames, numFeatures);
        if (returnCodeLGBM == -1) {
            logger.error("Could not set feature names.");
            throw new LightGBMException();
        }
    }

    /**
     * Creates the booster structure with all the parameters and training dataset resources (but doesn't train).
     *
     * @param swigTrainResources An object with the training resources already initialized.
     * @param trainParams        the LightGBM string-formatted string with properties in the form "key1=value1 key2=value2 ..."
     * @see LightGBMBinaryClassificationModelTrainer#trainBooster(SWIGTYPE_p_void, int) .
     */
    static void createBoosterStructure(final SWIGTrainResources swigTrainResources, final String trainParams) {

        logger.debug("Initializing LightGBM model structure.");
        final int returnCodeLGBM = lightgbmlib.LGBM_BoosterCreate(
                swigTrainResources.swigDatasetHandle,
                trainParams,
                swigTrainResources.swigOutBoosterHandlePtr
        );
        if (returnCodeLGBM == -1) {
            logger.error("LightGBM model structure creation failed.");
            throw new LightGBMException();
        }

        swigTrainResources.initSwigBoosterHandle();
    }

    /**
     * Train the LightGBM model from scratch given a dataset and an already existing BoosterHandle.
     * <p>
     * Trains by adding one iteration to the model, one at a time until it reaches the numIterations,
     * or receives a signal from the backend.
     *
     * @param swigBoosterHandle Handle to the boosting model
     * @param numIterations     Number of iterations to train (supports early stopping).
     */
    private static void trainBooster(final SWIGTYPE_p_void swigBoosterHandle, final int numIterations) {

        logger.info("Training LightGBM model.");
        final SWIGTYPE_p_int swigOutFinishedIntPtr = lightgbmlib.new_intp();

        try {
            for (int trainIteration = 0; trainIteration < numIterations; ++trainIteration) {
                logger.debug("Starting model training iteration #{}/{}.", trainIteration + 1, numIterations);
                final int returnCodeLGBM = lightgbmlib.LGBM_BoosterUpdateOneIter(swigBoosterHandle, swigOutFinishedIntPtr);
                if (returnCodeLGBM == -1) {
                    logger.error("Failed to train model!");
                    throw new LightGBMException();
                }

                if (lightgbmlib.intp_value(swigOutFinishedIntPtr) == 1) {
                    /*
                    Model signalled that all iterations are done, or early stopping was reached.
                     The latter could only happen if an in-train validation set was provided.
                    */
                    logger.info("LightGBM backend signalled the end of the model train.");
                    break;
                }
            }
        } finally {
            lightgbmlib.delete_intp(swigOutFinishedIntPtr);
        }

        logger.info("Finished model training.");
    }

    /**
     * Saves the model to disk.
     *
     * @param swigBoosterHandle SWIG booster handle
     * @param outputModelPath   output path where to save the model.
     */
    static void saveModelFileToDisk(final SWIGTYPE_p_void swigBoosterHandle, final Path outputModelPath) {

        logger.debug("Saving trained model to disk at {}.", outputModelPath);

        final int returnCodeLGBM = lightgbmlib.LGBM_BoosterSaveModel(
                swigBoosterHandle,
                0,
                -1,
                lightgbmlib.C_API_FEATURE_IMPORTANCE_GAIN,
                outputModelPath.toAbsolutePath().toString());
        if (returnCodeLGBM == -1) {
            logger.error("Could not save model to disk.");
            throw new LightGBMException();
        }

        logger.info("Saved model to disk");
    }

    /**
     * Takes the data in dataset and copies it into the features and label C++ arrays through SWIG.
     *
     * @param dataset              Input train dataset (with target label)
     * @param swigFeatureDataArray SWIG array of doubles to which the features data will be copied into (in row-major format).
     * @param swigLabelDataArray   SWIG array of floats to which the labels shall be copied.
     */
    private static void copyTrainDataToSWIGArrays(final Dataset dataset,
                                                  final SWIGTYPE_p_double swigFeatureDataArray,
                                                  final SWIGTYPE_p_float swigLabelDataArray) {

        final DatasetSchema datasetSchema = dataset.getSchema();
        final int numFields = datasetSchema.getFieldSchemas().size();
        final int numFeatures = datasetSchema.getPredictiveFields().size();
        // if target index doesn't exit, return -1.
        final int targetIndex = datasetSchema.getTargetIndex().orElse(-1);

        final Iterator<Instance> iterator = dataset.getInstances();
        int rowIdx = 0;
        while (iterator.hasNext()) {
            final Instance instance = iterator.next();

            // Set the label value for this instance:
            lightgbmlib.floatArray_setitem(swigLabelDataArray, rowIdx, (float) instance.getValue(targetIndex));

            // Set the features values for this instance:
            final long rowOffset = rowIdx * numFeatures;
            for (int colIdx = 0, afterTargetColOffset = 0; colIdx < numFields; ++colIdx) {
                if (colIdx == targetIndex) {
                    afterTargetColOffset = -1; // Initially 0, becomes -1 after passing the target column.
                } else {
                    lightgbmlib.doubleArray_setitem(
                            swigFeatureDataArray,
                            rowOffset + colIdx + afterTargetColOffset,
                            instance.getValue(colIdx)
                    );
                }
            }
            ++rowIdx;
        }
    }

    /**
     * Turns the List of parameters into a string with the parameters in the format LightGBM expects.
     *
     * @return LightGBM params string in the format LightGBM expects ("key1=value1 key2=value2 ... keyN=valueN").
     */
    private static String getLightGBMTrainParamsString(final Map<String, String> mapParams,
                                                       final List<Integer> categoricalFeatureIndices) {

        final ImmutableMap<String, String> allMapParams = ImmutableMap.<String, String>builder()
                .putAll(mapParams)
                .put("num_threads", "1")
                .put("application", "binary")
                .put("categorical_feature", StringUtils.join(categoricalFeatureIndices, ","))
                .build();

        final StringBuilder paramsBuilder = new StringBuilder();
        allMapParams.forEach((key, value) -> {
            paramsBuilder.append(String.format("%s=%s ", key, value));
        });

        return paramsBuilder.toString();
    }

}
