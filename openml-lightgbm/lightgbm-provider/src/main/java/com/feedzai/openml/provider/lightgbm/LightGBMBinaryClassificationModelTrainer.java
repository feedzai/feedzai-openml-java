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
import com.google.common.collect.ImmutableSet;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_int;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;
import java.util.HashMap;
import java.util.Optional;
import java.util.Set;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static com.feedzai.openml.provider.lightgbm.FairGBMDescriptorUtil.CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME;
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
     * Default number of instances per C++ train data chunk.
     * Train data is copied from the input stream into an array of chunks.
     * Each chunk will have this many instances. Must be set before `fit()`.
     * <p>
     * @implNote Performance overhead notes:
     * - Too small? Performance overhead - excessive in-memory data fragmentation.
     * - Too large? RAM overhead - in the worst case the last chunk has only 1 instance.
     *              Each instance might have upwards of 400 features. Each costs 8 bytes.
     *              E.g.: 100k instances of 400 features =>  320MB / chunk
     */
    static final long DEFAULT_TRAIN_DATA_CHUNK_INSTANCES_SIZE = 200000;

    /**
     * Placeholder for use when some integer argument is not provided.
     * E.g., when running standard unconstrained LightGBM the constrained_group_column parameter will take this value.
     */
    static final int NO_SPECIFIC = -1;

    /**
     * String prefix used to refer to columns by name in LightGBM configs/parameters.
     */
    static final String COL_NAME_PREFIX = "name:";


    /**
     * This class is not meant to be instantiated.
     */
    private LightGBMBinaryClassificationModelTrainer() {}

    /**
     * See LightGBMBinaryClassificationModelTrainer#fit overload below.
     * This version does not specify train data buffer chunk sizes to tune the memory layout.
     *
     * @param dataset             Train dataset.
     * @param params              LightGBM model parameters.
     * @param outputModelFilePath Output filepath for the model file in .txt format.
     */
    static void fit(final Dataset dataset,
                    final Map<String, String> params,
                    final Path outputModelFilePath) {

        fit(dataset,
            params,
            outputModelFilePath,
            DEFAULT_TRAIN_DATA_CHUNK_INSTANCES_SIZE);
    }

    /**
     * Train a LightGBM model from scratch, using streamed train data from a dataset iterator.
     * <p>
     * <b>Problem</b>
     * For performance reasons, the Dataset is read with a single-pass, however its size
     * is not known a priori.
     * <p>
     * <b>Solution</b>
     * A "ChunkedArray" is used in C++ to hold the streamed data before creating a LightGBM Dataset.
     * The ChunkedArray is a dynamic array of chunks (arrays), where all chunks have the same size.
     * <p>
     * Algorithm:
     * <ol>
     *     <li>Initialize ChunkedArray (starts with a single chunk).</li>
     *     <li>Add values from the input stream one by one to the end of the last chunk through array.add(value).</li>
     *     <li>When the chunk is full, ChunkedArray adds another chunk and inserts the value there.</li>
     *     <li>Repeat from 2 until the input is exhausted.</li>
     * </ol>
     * <p>
     * Computing the dataset size from the ChunkedArray is an O(1) operation.
     * As all chunks have the same size:
     * size = (num_chunks-1)*chunk_size + num_elements_in_last_chunk
     *
     * @param dataset             Train dataset.
     * @param params              LightGBM model parameters.
     * @param outputModelFilePath Output filepath for the model file in .txt format.
     * @param instancesPerChunk   Number of instances for each train chunk in C++.
     */
    static void fit(final Dataset dataset,
                    final Map<String, String> params,
                    final Path outputModelFilePath,
                    final long instancesPerChunk) {

        final DatasetSchema schema = dataset.getSchema();
        final int numFeatures = schema.getPredictiveFields().size();

        // Parse train parameters to LightGBM format
        final String trainParams = getLightGBMTrainParamsString(params, schema);

        final int numIterations = parseInt(params.get(NUM_ITERATIONS_PARAMETER_NAME));
        logger.debug("LightGBM model trainParams: {}", trainParams);

        final SWIGTrainData swigTrainData = new SWIGTrainData(
                numFeatures,
                instancesPerChunk,
                isFairnessConstrained(params));
        final SWIGTrainBooster swigTrainBooster = new SWIGTrainBooster();

        /// Create LightGBM dataset
        final int constraintGroupColIndex = getConstraintGroupColumnIndex(schema, params).orElse(NO_SPECIFIC);
        createTrainDataset(dataset, numFeatures, trainParams, constraintGroupColIndex, swigTrainData);

        /// Create Booster from dataset
        createBoosterStructure(swigTrainBooster, swigTrainData, trainParams);
        trainBooster(swigTrainBooster.swigBoosterHandle, numIterations);

        /// Save model
        saveModelFileToDisk(swigTrainBooster.swigBoosterHandle, outputModelFilePath);
        swigTrainBooster.close(); // Explicitly release C++ resources right away. They're no longer needed.
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
     * Gets the index of the constraint group column.
     * NOTE: the constraint group column must be part of the features in the Dataset, but it may be ignored for training
     *       (unawareness).
     * @param schema Schema of the dataset.
     * @param mapParams LightGBM train parameters.
     * @return the index of the constraint group column if one was provided, else returns an empty Optional.
     */
    private static Optional<Integer> getConstraintGroupColumnIndex(final DatasetSchema schema,
                                                                   final Map<String, String> mapParams) {
        // Parse the constraint_group_column, if one was provided
        String constraintGroupCol = mapParams.get(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME);
        if (constraintGroupCol == null || constraintGroupCol.trim().isEmpty()) {
            return Optional.empty();
        }

        // Trim white-space
        constraintGroupCol = constraintGroupCol.trim();

        // Initialize column index to placeholder
        int constraintGroupColIndex = NO_SPECIFIC;

        // Check if it's already in numeric format
        try {
            constraintGroupColIndex = Integer.parseInt(constraintGroupCol);
        }

        // If not, find the index for this column
        catch (NumberFormatException e) {

            // Remove the "name:" prefix
            final String actualConstraintGroupCol = constraintGroupCol.substring(
                    constraintGroupCol.startsWith(COL_NAME_PREFIX) ? COL_NAME_PREFIX.length() : 0);

            // Find index for this column; consider label offset (LightGBM indices disregard the target column)
            final List<FieldSchema> featureFields = schema.getPredictiveFields();
            Optional<FieldSchema> constraintGroupField = featureFields
                    .stream()
                    .filter(field -> field.getFieldName().equalsIgnoreCase(actualConstraintGroupCol))
                    .findFirst();

            // Check if column exists
            if (! constraintGroupField.isPresent()) {
                logger.error(String.format(
                        "The parameter %s=%s is invalid; no such column was found.",
                           CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME,
                           actualConstraintGroupCol));
                return Optional.empty();
            }

            // Check if the constraint_group_column is in categorical format
            if (! (constraintGroupField.get().getValueSchema() instanceof CategoricalValueSchema)) {
                logger.error(String.format(
                        "The parameter %s=%s is invalid; expected a column in categorical format, got %s format.",
                        CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME,
                        actualConstraintGroupCol,
                        constraintGroupField.get().getValueSchema().getClass().toString()));
                return Optional.empty();
            }

            // NOTE!
            //  - this index corresponds to the index in our dataset schema;
            //  - this value may be off by one when compared to LightGBM's expected index values;
            //  - this is due to the fact that LightGBM disregards the target column when counting column indices;
            constraintGroupColIndex = constraintGroupField.get().getFieldIndex();
        }

        return Optional.of(constraintGroupColIndex);
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
     * @param trainParams        LightGBM-formatted params string ("key1=value1 key2=value2 ...")
     * @param constraintGroupColIndex   The index of the constraint group column.
     * @param swigTrainData      SWIGTrainData object
     */
    private static void createTrainDataset(final Dataset dataset,
                                           final int numFeatures,
                                           final String trainParams,
                                           final int constraintGroupColIndex,
                                           final SWIGTrainData swigTrainData) {

        logger.info("Creating LightGBM dataset");

        logger.debug("Copying train data through SWIG.");
        copyTrainDataToSWIGArrays(
                dataset,
                swigTrainData,
                constraintGroupColIndex
        );

        initializeLightGBMTrainDatasetFeatures(
                swigTrainData,
                numFeatures,
                trainParams
        );

        setLightGBMDatasetLabelData(
                swigTrainData
        );

        if (constraintGroupColIndex != NO_SPECIFIC) {
            setLightGBMDatasetConstraintGroupData(
                    swigTrainData
            );
        }

        setLightGBMDatasetFeatureNames(swigTrainData.swigDatasetHandle, dataset.getSchema());

        logger.info("Created LightGBM dataset.");
    }

    /**
     * Initializes the LightGBM dataset structure and copies the feature data.
     *
     * @param swigTrainData SWIGTrainData object.
     * @param numFeatures   Number of features used to predict.
     * @param trainParams   LightGBM string with the train params ("key1=value1 key2=value2 ...").
     */
    private static void initializeLightGBMTrainDatasetFeatures(final SWIGTrainData swigTrainData,
                                                               final int numFeatures,
                                                               final String trainParams) {

        logger.debug("Initializing LightGBM in-memory structure and setting feature data.");

        /// First generate the array that has the chunk sizes for `LGBM_DatasetCreateFromMats`.
        final SWIGTYPE_p_int swigChunkSizesArray = genSWIGFeatureChunkSizesArray(swigTrainData, numFeatures);

        /// Now create the LightGBM Dataset itself from the chunks:
        logger.debug("Creating LGBM_Dataset from chunked data...");
        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetCreateFromMats(
                (int) swigTrainData.swigFeaturesChunkedArray.get_chunks_count(), // numChunks
                swigTrainData.swigFeaturesChunkedArray.data_as_void(), // input data (void**)
                lightgbmlibConstants.C_API_DTYPE_FLOAT64,
                swigChunkSizesArray,
                numFeatures,
                1, // rowMajor.
                trainParams, // parameters.
                null, // No alignment with other datasets.
                swigTrainData.swigOutDatasetHandlePtr // Output LGBM Dataset
        );
        if (returnCodeLGBM == -1) {
            logger.error("Could not create LightGBM dataset.");
            throw new LightGBMException();
        }

        swigTrainData.initSwigDatasetHandle();
        swigTrainData.destroySwigTrainFeaturesChunkedArray();
        lightgbmlib.delete_intArray(swigChunkSizesArray);
    }

    /**
     * Generates a SWIG array of ints with the size of each train chunk (partition).
     *
     * @param swigTrainData SWIGTrainData object.
     * @param numFeatures   Number of features used to predict.
     * @return SWIG (int*) array of the train chunks' sizes.
     */
    private static SWIGTYPE_p_int genSWIGFeatureChunkSizesArray(final SWIGTrainData swigTrainData,
                                                                final int numFeatures) {

        logger.debug("Retrieving chunked data block sizes...");

        final long numChunks = swigTrainData.swigFeaturesChunkedArray.get_chunks_count();
        final long chunkInstancesSize = swigTrainData.getNumInstancesChunk();
        final SWIGTYPE_p_int swigChunkSizesArray = lightgbmlib.new_intArray(numChunks);

        // All but the last chunk have the same size:
        for (int i = 0; i < numChunks - 1; ++i) {
            lightgbmlib.intArray_setitem(swigChunkSizesArray, i, (int) chunkInstancesSize);
            logger.debug("FTL: chunk-size report: chunk #{} is full-chunk of size {}", i, (int) chunkInstancesSize);
        }
        // The last chunk is usually partially filled:
        lightgbmlib.intArray_setitem(
                swigChunkSizesArray,
                numChunks - 1,
                (int) swigTrainData.swigFeaturesChunkedArray.get_last_chunk_add_count() / numFeatures
        );
        logger.debug("FTL: chunk-size report: chunk #{} is partial-chunk of size {}",
                numChunks - 1,
                (int) swigTrainData.swigFeaturesChunkedArray.get_last_chunk_add_count() / numFeatures);

        return swigChunkSizesArray;
    }

    /**
     * Sets the LightGBM dataset label data.
     *
     * @param swigTrainData SWIGTrainData object.
     */
    private static void setLightGBMDatasetLabelData(final SWIGTrainData swigTrainData) {

        final long numInstances = swigTrainData.swigLabelsChunkedArray.get_add_count();
        swigTrainData.initSwigTrainLabelDataArray(); // Init and copy from chunked data.
        logger.debug("FTL: #labels={}", numInstances);

        logger.debug("Setting label data.");
        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetSetField(
                swigTrainData.swigDatasetHandle,
                "label", // LightGBM label column type.
                lightgbmlib.float_to_voidp_ptr(swigTrainData.swigTrainLabelDataArray),
                (int) numInstances,
                lightgbmlibConstants.C_API_DTYPE_FLOAT32
        );
        if (returnCodeLGBM == -1) {
            logger.error("Could not set label.");
            throw new LightGBMException();
        }

        swigTrainData.destroySwigTrainLabelDataArray();
    }

    /**
     * Sets the LightGBM dataset fairness constraint group data.
     *
     * @param swigTrainData SWIGTrainData object.
     */
    private static void setLightGBMDatasetConstraintGroupData(final SWIGTrainData swigTrainData) {

        final long numInstances = swigTrainData.swigConstraintGroupChunkedArray.get_add_count();
        swigTrainData.initSwigConstraintGroupDataArray(); // Init and copy from chunked data.
        logger.debug("FTL: #labels={}", numInstances);

        logger.debug("Setting constraint group data.");
        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetSetField(
                swigTrainData.swigDatasetHandle,
                "constraint_group", // LightGBM label column type.
                lightgbmlib.int_to_voidp_ptr(swigTrainData.swigConstraintGroupDataArray),
                (int) numInstances,
                lightgbmlibConstants.C_API_DTYPE_INT32
        );
        if (returnCodeLGBM == -1) {
            logger.error("Could not set constraint group data.");
            throw new LightGBMException();
        }

        swigTrainData.destroySwigConstraintGroupDataArray();
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
     * @param swigTrainBooster An object with the training resources already initialized.
     * @param swigTrainData      SWIGTrainData object.
     * @param trainParams        the LightGBM string-formatted string with properties in the form "key1=value1 key2=value2 ...".
     * @see LightGBMBinaryClassificationModelTrainer#trainBooster(SWIGTYPE_p_void, int) .
     */
    static void createBoosterStructure(final SWIGTrainBooster swigTrainBooster,
                                       final SWIGTrainData swigTrainData,
                                       final String trainParams) {

        logger.debug("Initializing LightGBM model structure.");
        final int returnCodeLGBM = lightgbmlib.LGBM_BoosterCreate(
                swigTrainData.swigDatasetHandle,
                trainParams,
                swigTrainBooster.swigOutBoosterHandlePtr
        );

        if (returnCodeLGBM == -1) {
            logger.error("LightGBM model structure creation failed.");
            throw new LightGBMException();
        }

        swigTrainBooster.initSwigBoosterHandle();
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
     * @param dataset       Input train dataset (with target label)
     * @param swigTrainData SWIGTrainData object.
     */
    private static void copyTrainDataToSWIGArrays(final Dataset dataset,
                                                  final SWIGTrainData swigTrainData) {
        copyTrainDataToSWIGArrays(dataset, swigTrainData, NO_SPECIFIC);
    }

    private static void copyTrainDataToSWIGArrays(final Dataset dataset,
                                                  final SWIGTrainData swigTrainData,
                                                  final int constraintGroupIndex) {

        final DatasetSchema datasetSchema = dataset.getSchema();
        final int numFields = datasetSchema.getFieldSchemas().size();
        /* Supervised model. Target must be present.
           This is ensured in validateForFit, by using the
           ValidationUtils' validateCategoricalSchema:
         */
        final int targetIndex = datasetSchema.getTargetIndex().get();

        final Iterator<Instance> iterator = dataset.getInstances();
        while (iterator.hasNext()) {
            final Instance instance = iterator.next();

            swigTrainData.addLabelValue((float) instance.getValue(targetIndex));

            if (constraintGroupIndex != NO_SPECIFIC) {
                swigTrainData.addConstraintGroupValue((int) instance.getValue(constraintGroupIndex));
            }

            for (int colIdx = 0; colIdx < numFields; ++colIdx) {
                if (colIdx != targetIndex) {
                    swigTrainData.addFeatureValue(instance.getValue(colIdx));
                }
            }
        }
        logger.debug("Copied train data of size {} into {} chunks.",
                swigTrainData.swigLabelsChunkedArray.get_add_count(),
                swigTrainData.swigLabelsChunkedArray.get_chunks_count()
        );
        if (swigTrainData.swigLabelsChunkedArray.get_add_count() == 0) {
            logger.error("Received empty train dataset!");
            throw new IllegalArgumentException("Received empty train dataset for LightGBM!");
        }
    }

    /**
     * Turns the List of parameters into a string with the parameters in the format LightGBM expects.
     *
     * @return LightGBM params string in the format LightGBM expects ("key1=value1 key2=value2 ... keyN=valueN").
     */
    private static String getLightGBMTrainParamsString(final Map<String, String> mapParams,
                                                       final DatasetSchema schema) {
        final Map<String, String> preprocessedMapParams = new HashMap<>();

        // Add categorical_feature parameter
        List<Integer> categoricalFeatureIndices = getCategoricalFeaturesIndicesWithoutLabel(schema);
        preprocessedMapParams
                .put("categorical_feature", StringUtils.join(categoricalFeatureIndices, ","));

        // Add constraint_group_column parameter
        final Optional<Integer> constraintGroupColumnIndex = getConstraintGroupColumnIndex(schema, mapParams);
        if (mapParams.containsKey(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME) && constraintGroupColumnIndex.isPresent()) {

            // NOTE! LightGBM counts column indices disregarding the target column
            final int fieldAfterLabelOffset = constraintGroupColumnIndex.get() > schema.getTargetIndex().get() ? -1 : 0;
            preprocessedMapParams.put(
                    CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME,
                    Integer.toString(constraintGroupColumnIndex.get() + fieldAfterLabelOffset));
        }

        // Set default objective parameter
        Optional<String> objective = getLightGBMObjective(mapParams);
        if (! objective.isPresent()) {
            // Default to objective=binary
            preprocessedMapParams.put("objective", "binary");
        }

        // Add all **other** parameters
        mapParams.forEach(preprocessedMapParams::putIfAbsent);

        // Build string containing params in LightGBM format
        final StringBuilder paramsBuilder = new StringBuilder();
        preprocessedMapParams.forEach((key, value) -> paramsBuilder.append(String.format("%s=%s ", key, value)));

        return paramsBuilder.toString();
    }

    /**
     * Gets the objective function within the given LightGBM train parameters.
     *
     * @param mapParams the LightGBM train parameters.
     * @return the objective function if one exists, else returns null.
     */
    public static Optional<String> getLightGBMObjective(final Map<String, String> mapParams) {
        // All aliases for the "objective" parameter https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective
        Set<String> objectiveAliases = ImmutableSet.of("objective", "objective_type", "app", "application", "loss");

        // Check whether the "objective" parameter was provided; if not, use the default value of "objective=binary"
        Map.Entry<String, String> candidate = mapParams
                .entrySet().stream()
                .filter(entry -> objectiveAliases.contains(entry.getKey()))
                .findFirst().orElse(null);

        return candidate != null ? Optional.of(candidate.getValue()) : Optional.empty();
    }

    /**
     * Whether the given mapParams correspond to a constrained LightGBM objective (aka FairGBM).
     * @param mapParams set of train parameters for LightGBM.
     * @return whether it is a fairness constrained objective.
     */
    public static boolean isFairnessConstrained(final Map<String, String> mapParams) {
        Optional<String> objective = getLightGBMObjective(mapParams);
        return (objective.isPresent() && objective.get().startsWith("constrained_"));
    }
}
