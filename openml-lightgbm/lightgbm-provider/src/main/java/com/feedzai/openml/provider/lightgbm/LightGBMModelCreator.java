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
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.provider.model.MachineLearningModelTrainer;
import com.feedzai.openml.util.load.LoadSchemaUtils;
import com.google.common.collect.ImmutableList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
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
 * Loads and/or fits LightGBM models.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
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
    static final String ERROR_MSG_CANNOT_LOAD_NON_BINARY_LIGHTGBM_MODEL = "Cannot load a non-binary LightGBM model.";

    /**
     * Error message for when using a schema with string fields.
     */
    static final String ERROR_MSG_SCHEMA_HAS_STRING_FIELDS = "Schema has string fields.";

    /**
     * Error message for when the target is not binary.
     */
    static final String ERROR_MSG_NON_BINARY_TARGET = "Target field must be binary.";

    /**
     * Error message prefix for when the model resource cannot be found.
     */
    static final String ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE = "Cannot find model file";

    /**
     * Error message when schema doesn't have the correct number of predictive fields.
     */
    static final String ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELDS_SIZE =
            "Received schema with wrong number of predictive fields.";

    /**
     * Error message when schema predictive fields don't match model feature names.
     */
    static final String ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELD_NAMES =
            "Received schema with wrong predictive field names.";

    /**
     * Error message when boosting type is Random Forest and bagging is not enabled.
     */
    static final String ERROR_MSG_RANDOM_FOREST_REQUIRES_BAGGING =
            "Random Forest Boosting type requires bagging. Please see bagging parameters.";

    /**
     * Constructor.
     * Must load the libraries so that the rest of the classes work.
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
            LightGBMBinaryClassificationModelTrainer.fit(
                    dataset, params, tmpModelFilePath);

            // Build a schema without the weight column for model loading validation
            final DatasetSchema schemaForLoading = getSchemaForLoading(params, dataset);
            return loadModel(tmpModelFilePath, schemaForLoading);
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

    /**
     * Retrieves the appropriate schema to be used to load a LightGBM model.
     * If the model was trained with sample weight, then its schema will consist in the train
     * data schema without the sample weight column; otherwise, the original raw schema of
     * the train data will be considered.
     *
     * @param params   LightGBM model parameters.
     * @param dataset  Train dataset.
     * @return         A {@link DatasetSchema} instance excluding the sample weight field if present;
     *                 otherwise, the original schema layout mapped to the dataset
     */
    private DatasetSchema getSchemaForLoading(Map<String, String> params, Dataset dataset) {
        final DatasetSchema schemaForLoading;

        final Optional<Integer> weightColIdx =
                SampleWeightParamParserUtil.getSampleWeightColumnIndex(params, dataset.getSchema());
        if (weightColIdx.isPresent()) {
            final List<FieldSchema> fieldsWithoutWeight = dataset.getSchema().getPredictiveFields().stream()
                                                                 .filter(field -> field.getFieldIndex() != weightColIdx.get())
                                                                 .collect(Collectors.toList());
            schemaForLoading = new DatasetSchema(fieldsWithoutWeight);
        } else {
            schemaForLoading = dataset.getSchema();
        }
        return schemaForLoading;
    }

    @Override
    public List<ParamValidationError> validateForFit(final Path pathToPersist,
                                                     final DatasetSchema schema,
                                                     final Map<String, String> params) {
        final ImmutableList.Builder<ParamValidationError> errorsBuilder = ImmutableList.builder();

        errorsBuilder
                .addAll(validateModelPathToTrain(pathToPersist))
                .addAll(validateSchema(schema))
                .addAll(validateFitParams(params))
                .addAll(validateSampleWeightsCol(schema, params));

        return errorsBuilder.build();
    }

    /**
     * Validate model fit schema.
     *
     * @param schema schema to validate
     * @return list of validation errors.
     */
    private List<ParamValidationError> validateSchema(final DatasetSchema schema) {

        final ImmutableList.Builder<ParamValidationError> errorsBuilder = ImmutableList.builder();

        validateCategoricalSchema(schema).ifPresent(errorsBuilder::add);

        if (schemaHasStringFields(schema)) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_SCHEMA_HAS_STRING_FIELDS));
        }

        if (getNumTargetClasses(schema).orElse(-1) != 2) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_NON_BINARY_TARGET));
        }

        return errorsBuilder.build();
    }

    /**
     * Ensure that if the sample weight column parameter is specified, it corresponds
     * to a column in the dataset schema
     *
     * @param schema Dataset schema.
     * @param params Model fit parameters.
     * @return list of validation errors.
     */
    private List<ParamValidationError> validateSampleWeightsCol(DatasetSchema schema,
                                                                final Map<String, String> params) {
        // Don't test anything if the parameter is not set:
        final Optional<String> sampleWeightFieldName = SampleWeightParamParserUtil.getSampleWeightFieldName(params);
        if (!sampleWeightFieldName.isPresent()) {
            return ImmutableList.of();
        }

        // Check if the field exists in the dataset:
        final Optional<Integer> sampleWeightColIndex =
                SampleWeightParamParserUtil.getSampleWeightColumnIndex(params, schema);
        if (!sampleWeightColIndex.isPresent()) {
            return ImmutableList.of(new ParamValidationError(String.format(
                    "Sample weight field %s doesn't exist in the dataset.",
                    sampleWeightFieldName.get()
            )));
        }

        // Ensure the sample weight field is numeric
        final FieldSchema sampleWeightSchema = schema.getFieldSchemas().get(sampleWeightColIndex.get());
        final AbstractValueSchema valueSchema = sampleWeightSchema.getValueSchema();
        if (!(valueSchema instanceof NumericValueSchema)) {
            return ImmutableList.of(new ParamValidationError("Sample weight must be a numeric field!"));
        }

        return ImmutableList.of();
    }

    /**
     * Validate model fit parameters.
     *
     * @param params Model fit parameters
     * @return list of validation errors.
     */
    private List<ParamValidationError> validateFitParams(final Map<String, String> params) {

        final ImmutableList.Builder<ParamValidationError> errorsBuilder = ImmutableList.builder();

        errorsBuilder.addAll(checkParams(
                LightGBMAlgorithms.LIGHTGBM_BINARY_CLASSIFIER.getAlgorithmDescriptor(), params));

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

        final Path modelFilePath = getPath(modelPath);

        logger.info("Loading LightGBM model from " + modelFilePath.toAbsolutePath());
        final LightGBMBinaryClassificationModel model = new LightGBMBinaryClassificationModel(
                modelFilePath, schema
        );

        // LightGBM considers binary classification as a special case of 1 class:
        if (!model.isModelBinary()) {
            throw new ModelLoadingException(ERROR_MSG_CANNOT_LOAD_NON_BINARY_LIGHTGBM_MODEL);
        }

        // Check predictive fields size
        //  - In LightGBM, if the sample weights are provided then the trained model will not
        //    have this field, but the DatasetSchema will -- Need to exclude this field
        //    In short, ensure all fields in model.getBoosterNumFeatures() exist in the DatasetSchema
        final String[] boosterFeatureNames = model.getBoosterFeatureNames();
        final Set<String> modelFeatureSet = new HashSet<>(Arrays.asList(boosterFeatureNames));
        final List<FieldSchema> relevantFields = schema.getPredictiveFields().stream()
                .filter(field -> modelFeatureSet.contains(field.getFieldName().replace(" ", "_")))
                .collect(Collectors.toList());

        if (relevantFields.size() != boosterFeatureNames.length) {
            throw new ModelLoadingException(ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELDS_SIZE);
        }

        if (!schemaMatchAllFeatures(relevantFields, boosterFeatureNames)) {
            throw new ModelLoadingException(ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELD_NAMES);
        }

        return model;
    }

    /**
     * Gets the model {@link Path}. If modelPath is a directory, gets the model file inside.
     *
     * @param modelPath The model {@link Path}.
     * @return The model file path.
     * @since 1.3.0
     */
    private Path getPath(final Path modelPath) {

        if (Files.isDirectory(modelPath)) {
            return modelPath.resolve(MODEL_BINARY_RESOURCE_FILE_NAME);
        }

        return modelPath;
    }

    @Override
    public List<ParamValidationError> validateForLoad(final Path modelPath,
                                                      final DatasetSchema schema,
                                                      final Map<String, String> params) {

        final ImmutableList.Builder<ParamValidationError> errorsBuilder = ImmutableList.builder();

        errorsBuilder.addAll(baseLoadValidations(schema, params));
        validateCategoricalSchema(schema).ifPresent(errorsBuilder::add);

        if (schemaHasStringFields(schema)) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_SCHEMA_HAS_STRING_FIELDS));
        }

        if (getNumTargetClasses(schema).orElse(-1) != 2) {
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_NON_BINARY_TARGET));
        }

        if (!Files.exists(modelPath)) {
            logger.error(ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE + " in filesystem ({}).", modelPath);
            errorsBuilder.add(new ParamValidationError(ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE + " in filesystem."));

            return errorsBuilder.build();
        }

        if (Files.isDirectory(modelPath) && !Files.exists(modelPath.resolve(MODEL_BINARY_RESOURCE_FILE_NAME))) {
            logger.error(
                    "Error loading model from directory ({}). File {} not found.",
                    modelPath, MODEL_BINARY_RESOURCE_FILE_NAME
            );
            errorsBuilder.add(new ParamValidationError(String.format("%s %s inside folder.",
                    ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE, MODEL_BINARY_RESOURCE_FILE_NAME)
            ));
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
        }
        return Optional.empty();
    }

    /**
     * Gets the feature names from the given fields.
     *
     * @implNote The space character is replaced with underscore
     * to comply with LightGBM's model features representation.
     *
     * @param fields List of field schemas.
     * @return Feature names from the schema.
     * @since 1.0.18
     */
    private static String[] getFeatureNamesFrom(final List<FieldSchema> fields) {

        return fields.stream()
                .map(FieldSchema::getFieldName)
                .map(fieldName -> fieldName.replace(" ", "_"))
                .toArray(String[]::new);
    }

    /**
     * Performs a one-by-one feature name comparison between a
     * given list of {@link FieldSchema} and an array of feature
     * names. This way the first mismatch is logged, improving debug.
     *
     * @param schemaRelevantFeatureNames Schema
     * @param featureNames Feature names to validate.
     * @return {@code true} if the schema predictive field names
     * match the provided array, {@code false} otherwise.
     * @since 1.0.18
     */
    private boolean schemaMatchAllFeatures(List<FieldSchema> schemaRelevantFeatureNames,
                                           final String[] featureNames) {

        final String[] schemaFeatureNames = getFeatureNamesFrom(schemaRelevantFeatureNames);

        boolean isMatch = true;

        for (int i = 0; i < featureNames.length; i++) {

            if (!schemaFeatureNames[i].equals(featureNames[i])) {

                logger.error("Schema with wrong predictive field name at index {}: '{}' Expected: '{}'",
                        i,
                        schemaFeatureNames[i],
                        featureNames[i]);

                isMatch = false;
            }
        }

        if (!isMatch) {
            logger.error("Schema with wrong predictive field names: '{}' - Expected: '{}'",
                    String.join(", ", schemaFeatureNames),
                    String.join(", ", featureNames));
        }

        return isMatch;
    }
}
