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
import com.feedzai.openml.data.schema.AbstractValueSchema;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.provider.model.MachineLearningModelTrainer;
import com.feedzai.openml.util.load.LoadSchemaUtils;
import com.google.common.collect.ImmutableList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.BAGGING_FRACTION_PARAMETER_NAME;
import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.BAGGING_FREQUENCY_PARAMETER_NAME;
import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.BOOSTING_TYPE_PARAMETER_NAME;
import static com.feedzai.openml.util.validate.ValidationUtils.baseLoadValidations;
import static com.feedzai.openml.util.validate.ValidationUtils.checkParams;
import static com.feedzai.openml.util.validate.ValidationUtils.validateCategoricalSchema;
import static com.feedzai.openml.util.validate.ValidationUtils.validateModelPathToTrain;
import static java.nio.file.Files.createTempFile;

/**
 * Loads the scoring model.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 */
public class LightGBMModelCreator implements MachineLearningModelTrainer<LightGBMBinaryClassificationModel> {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(LightGBMModelCreator.class);

    /**
     * LightGBM model resource file name inside the mode folder, when model is saved/loaded in/from directory.
     */
    public static final String MODEL_BINARY_RESOURCE_FILE_NAME = "LightGBM_model.txt";

    /**
     * Error message thrown for when trying to load a non-binary model.
     */
    public static final String ERROR_MSG_CANNOT_LOAD_NON_BINARY_LIGHTGBM_MODEL = "Cannot load a non-binary LightGBM model.";

    /**
     * Error message for when using a schema with string fields.
     */
    public static final String ERROR_MSG_SCHEMA_HAS_STRING_FIELDS = "Schema has string fields.";

    /**
     * Error message for when the target is not binary.
     */
    public static final String ERROR_MSG_NON_BINARY_TARGET = "Target field must be binary.";

    /**
     * Error message prefix for when the model resource cannot be found.
     */
    public static final String ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE = "Cannot find model file";

    /**
     * Error message when schema doesn't have the correct number of predictive fields.
     */
    public static final String ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELDS_SIZE =
            "Received schema with wrong number of predictive fields.";

    /**
     * Error message when boosting type is Random Forest and bagging is not enabled.
     */
    public static final String ERROR_MSG_RANDOM_FOREST_REQUIRES_BAGGING =
            "Random Forest Boosting type requires bagging. Please see bagging parameters.";

    /**
     * Constructor.
     * Its only job is to load the libraries so that the rest of the classes work.
     * Without the libraries instantiated, not even LightGBM exceptions can be thrown.
     */
    public LightGBMModelCreator() {
        // Initialize libs. Before this call, no lightgbmlib* methods can be called:
        LightGBMUtils.loadLibs();
    }


    @Override
    public LightGBMBinaryClassificationModel fit(final Dataset dataset,
                                                 final Random random,
                                                 final Map<String, String> params) {

        final Path tmpModelFilePath;

        try {
            tmpModelFilePath = createTempFile("pulse_lightgbm_model_", null);
        } catch (final IOException e) {
            logger.error("Could not create temporary file.");
            throw new RuntimeException(e);
        }

        try {
            LightGBMBinaryClassificationModelTrainer.fit(dataset, params, tmpModelFilePath);
            return loadModel(tmpModelFilePath, dataset.getSchema());
        } catch (final Exception e) {
            logger.error("Could not train the model.");
            throw new RuntimeException(e);
        } finally {
            try {
                Files.delete(tmpModelFilePath);
            } catch (final IOException e) {
                logger.error("Could not delete temporary model file: {}", e.getMessage());
            }
        }
    }

    @Override
    public List<ParamValidationError> validateForFit(final Path pathToPersist,
                                                     final DatasetSchema schema,
                                                     final Map<String, String> params) {
        final ImmutableList.Builder<ParamValidationError> errorsBuilder = ImmutableList.builder();

        errorsBuilder.addAll(validateModelPathToTrain(pathToPersist));
        errorsBuilder.addAll(checkParams(
                LightGBMAlgorithms.LIGHTGBM_BINARY_CLASSIFIER.getAlgorithmDescriptor(), params));
        validateCategoricalSchema(schema).ifPresent(errorsBuilder::add);

        if (schemaHasStringFields(schema)) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_SCHEMA_HAS_STRING_FIELDS));
        }

        if (getNumTargetClasses(schema).orElse(-1) != 2) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_NON_BINARY_TARGET));
        }

        if (params.get(BOOSTING_TYPE_PARAMETER_NAME).equals("rf") && baggingDisabled(params)) {
            logger.warn("RF requires bagging. Set bagging fraction < 1 and bagging frequency > 0.");
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_RANDOM_FOREST_REQUIRES_BAGGING));
        }

        return errorsBuilder.build();
    }

    /**
     * Checks if bagging is disabled.
     *
     * @param params LightGBM parameters
     * @return true if disabled, false otherwise.
     */
    private boolean baggingDisabled(final Map<String, String> params) {

        final double epsilon = 1e-60;

        final double freq = Double.parseDouble(params.get(BAGGING_FREQUENCY_PARAMETER_NAME));
        final double fraction = Double.parseDouble(params.get(BAGGING_FRACTION_PARAMETER_NAME));
        return ((Math.abs(freq - 0) < epsilon) || (Math.abs(1 - fraction) < epsilon));
    }

    @Override
    public LightGBMBinaryClassificationModel loadModel(final Path modelPath,
                                                       final DatasetSchema schema)  throws ModelLoadingException {

        // If modelPath is a directory, look for the model file inside:
        Path modelFilePath = modelPath;
        if (Files.isDirectory(modelPath)) {
            modelFilePath = modelPath.resolve(MODEL_BINARY_RESOURCE_FILE_NAME);
        }

        logger.info("Loading LightGBM model from " + modelFilePath.toAbsolutePath().toString());
        final LightGBMBinaryClassificationModel model = new LightGBMBinaryClassificationModel(
                modelFilePath, schema
        );

        // LightGBM considers binary classification as a special case of 1 class:
        if (! model.isModelBinary()) {
            throw new ModelLoadingException(ERROR_MSG_CANNOT_LOAD_NON_BINARY_LIGHTGBM_MODEL);
        }

        if (model.getBoosterNumFeatures() != schema.getPredictiveFields().size()) {
            throw new ModelLoadingException(ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELDS_SIZE);
        }

        /*
          TODO: https://issues.feedzai.com/browse/PULSEDEV-30615 CHECK FEATURE NAMES ORDER IN MODEL AGAINST LOADED SCHEMA.
           Currently there is a bug in the lightgbmlibJNI interface (lightgbmlib.i) specification which prematurely deallocates
           the memory before use (%array_functions(char *, stringArray)).
           This means we cannot implement this unless we add a new method to the interface to correctly allocate and
           free memory for the char*, and even char+* case (swigArrayOfStrings):
            int retcode = lightgbmlibJNI.LGBM_BoosterGetFeatureNames(swigBoosterHandle, this.swigOutLengthIntPtr, swigArrayOfStrings);
            UPDATE:
             Already implemented lightgbm custom code for that. See https://issues.feedzai.com/browse/PULSEDEV-30615.
        */

        return model;
    }

    @Override
    public List<ParamValidationError> validateForLoad(final Path modelPath,
                                                      final DatasetSchema schema,
                                                      final Map<String, String> params) {

        final ImmutableList.Builder<ParamValidationError> errorsBuilder = ImmutableList.builder();

        if (Files.isDirectory(modelPath)) {
            if (! Files.exists(modelPath.resolve(MODEL_BINARY_RESOURCE_FILE_NAME))) {
                logger.error(
                        "Error loading model from directory ({}). File {} not found.",
                        modelPath, MODEL_BINARY_RESOURCE_FILE_NAME
                );
                errorsBuilder.add(new ParamValidationError(String.format("%s %s inside folder.",
                        ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE, MODEL_BINARY_RESOURCE_FILE_NAME)
                ));
            }
        } else if (! Files.exists(modelPath)) {
            logger.error(ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE + " in filesystem ({}).", modelPath);
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE + " in filesystem."));
        }

        // Checks that the RandomForest also implements:
        errorsBuilder.addAll(baseLoadValidations(schema, params));
        validateCategoricalSchema(schema).ifPresent(errorsBuilder::add);

        if (schemaHasStringFields(schema)) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_SCHEMA_HAS_STRING_FIELDS));
        }

        if (getNumTargetClasses(schema).orElse(-1) != 2) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_NON_BINARY_TARGET));
        }

        return errorsBuilder.build();
    }

    @Override
    public DatasetSchema loadSchema(final Path modelPath) throws ModelLoadingException {

        return LoadSchemaUtils.datasetSchemaFromJson(modelPath);
    }

    /**
     * Checks if schema has string fields.
     *
     * @param schema schema
     * @return boolean which is true if there are string fields
     */
    private static boolean schemaHasStringFields(final DatasetSchema schema) {

        return schema.getFieldSchemas().stream().anyMatch(field -> field.getValueSchema() instanceof StringValueSchema);
    }

    /**
     * Gets the number of target classes in the schema target fields.
     *
     * @param schema Schema
     * @return Number of target classes in the schema target
     * field or empty if there is no target field, or is not binary.
     */
    private static Optional<Integer> getNumTargetClasses(final DatasetSchema schema) {

        if (schema.getTargetFieldSchema().isPresent()) {
            final AbstractValueSchema fieldValueSchema = schema.getTargetFieldSchema().get().getValueSchema();
            if (fieldValueSchema instanceof CategoricalValueSchema) {
                return Optional.of(((CategoricalValueSchema) fieldValueSchema).getNominalValues().size());
            } else {
                return Optional.empty();
            }
        } else {
            return Optional.empty();
        }
    }
}
