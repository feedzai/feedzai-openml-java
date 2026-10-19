/*
 * Copyright 2026 Feedzai
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

package com.feedzai.openml.provider.xgboost;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.feedzai.openml.provider.model.MachineLearningModelTrainer;
import com.feedzai.openml.util.load.LoadModelUtils;
import com.feedzai.openml.util.load.LoadSchemaUtils;
import com.feedzai.openml.util.validate.ValidationUtils;
import com.google.common.collect.ImmutableList;
import com.google.common.io.MoreFiles;
import com.google.common.io.RecursiveDeleteOption;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

/**
 * Loads and trains XGBoost models through the native {@code xgboost4j} JVM package.
 *
 * <p><b>Training is fully in-process (no Spark)</b>, mirroring how the H2O provider trains inside an
 * embedded in-JVM instance: the dataset is materialized into an in-memory {@link DMatrix},
 * {@link XGBoost#train} builds the booster, the booster is exported and then reloaded. Because
 * {@code xgboost4j} is a thin JNI wrapper it runs on Java 8-25 and on ARM (Graviton / Apple Silicon).
 */
public class XgboostModelCreator implements MachineLearningModelTrainer<XgboostClassificationModel> {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(XgboostModelCreator.class);

    /**
     * Name of the model file written inside the model folder, using XGBoost's portable UBJSON format.
     */
    public static final String MODEL_BINARY_RESOURCE_FILE_NAME = "XGBoost_model.ubj";

    /**
     * Prefix for the temporary directory holding a freshly trained model before it is reloaded.
     */
    private static final String EXPORT_DIR_PREFIX = "fdz_xgboost_";

    /**
     * Value used to signal a missing feature to XGBoost.
     */
    private static final float MISSING_VALUE = Float.NaN;

    /**
     * Default number of boosting rounds used if the parameter is absent. Matches the default exposed in
     * {@link XgboostDescriptorUtil} and the previous (DS Python package) implementation.
     */
    private static final int DEFAULT_NUM_ROUND = 50;

    /**
     * Error reported when the schema contains free-text (string) fields, which XGBoost cannot consume.
     */
    static final String ERROR_MSG_SCHEMA_HAS_STRING_FIELDS =
            "XGBoost only supports numeric and categorical fields; the schema contains string fields.";

    /**
     * Error reported when the target field is not binary. This provider only offers binary classification.
     */
    static final String ERROR_MSG_NON_BINARY_TARGET =
            "XGBoost is offered here as a binary classifier; the target field must have exactly two classes.";

    /**
     * Error reported when the loaded model's feature count does not match the schema's predictive fields.
     */
    static final String ERROR_MSG_FEATURE_COUNT_MISMATCH =
            "The XGBoost model expects %d features but the provided schema describes %d predictive fields.";

    @Override
    public XgboostClassificationModel loadModel(final Path modelPath, final DatasetSchema schema)
            throws ModelLoadingException {

        logger.info("Loading XGBoost model from [{}].", modelPath);
        final String modelFilePath = resolveModelFile(modelPath).toAbsolutePath().toString();

        try {
            final Booster booster = XGBoost.loadModel(modelFilePath);

            // Fail fast (parity with the LightGBM provider) if the model's feature count does not match the
            // schema. Without this, a mismatched schema loads "successfully" and only blows up later inside
            // inplace_predict at scoring time, with a far less actionable error.
            final int expectedFeatures = XgboostSchemaUtils.numFeatures(schema);
            final long modelFeatures = booster.getNumFeature();
            if (modelFeatures != expectedFeatures) {
                booster.dispose();
                throw new ModelLoadingException(
                        String.format(ERROR_MSG_FEATURE_COUNT_MISMATCH, modelFeatures, expectedFeatures));
            }

            logger.info("XGBoost model loaded successfully.");
            return new XgboostClassificationModel(booster, schema);
        } catch (final XGBoostError e) {
            throw new ModelLoadingException(
                    String.format("Failed to load the XGBoost model from [%s].", modelFilePath), e);
        }
    }

    /**
     * Resolves the actual model file to load, supporting both layouts used across the codebase:
     * <ul>
     *     <li>the Pulse import / training-export layout {@code <dir>/model/<file>};</li>
     *     <li>a model file written directly at the root of a directory (as produced by
     *     {@link XgboostClassificationModel#save(Path, String)});</li>
     *     <li>a direct path to the model file itself.</li>
     * </ul>
     *
     * @param modelPath The path provided to {@link #loadModel(Path, DatasetSchema)}.
     * @return The path of the model file to load.
     * @throws ModelLoadingException If the model file cannot be located within the model folder layout.
     */
    static Path resolveModelFile(final Path modelPath) throws ModelLoadingException {
        if (!Files.isDirectory(modelPath)) {
            return modelPath;
        }
        if (Files.isDirectory(modelPath.resolve(LoadModelUtils.MODEL_FOLDER))) {
            return LoadModelUtils.getModelFilePath(modelPath);
        }
        return modelPath.resolve(MODEL_BINARY_RESOURCE_FILE_NAME);
    }

    @Override
    public DatasetSchema loadSchema(final Path modelPath) throws ModelLoadingException {
        return LoadSchemaUtils.datasetSchemaFromJson(modelPath);
    }

    @Override
    public List<ParamValidationError> validateForLoad(final Path modelPath,
                                                      final DatasetSchema schema,
                                                      final Map<String, String> params) {
        final ImmutableList.Builder<ParamValidationError> errorBuilder = ImmutableList.builder();

        errorBuilder.addAll(ValidationUtils.baseLoadValidations(schema, params));
        errorBuilder.addAll(ValidationUtils.validateModelInDir(modelPath));
        ValidationUtils.validateCategoricalSchema(schema).ifPresent(errorBuilder::add);
        if (schemaHasStringFields(schema)) {
            errorBuilder.add(new ParamValidationError(ERROR_MSG_SCHEMA_HAS_STRING_FIELDS));
        }

        return errorBuilder.build();
    }

    @Override
    public XgboostClassificationModel fit(final Dataset dataset,
                                          final Random random,
                                          final Map<String, String> params) throws ModelTrainingException {

        final DatasetSchema schema = dataset.getSchema();

        DMatrix trainMatrix = null;
        Booster booster = null;
        Path exportDir = null;
        try {
            // Parse/translate the (cheap) parameters before building the (expensive, native) DMatrix so a
            // malformed value such as a non-numeric num_round fails fast and is surfaced as a
            // ModelTrainingException rather than a raw NumberFormatException escaping native work.
            final Map<String, Object> boosterParams = toBoosterParams(params, random);
            final int numRound = numRoundOf(params);

            trainMatrix = buildTrainMatrix(dataset);

            booster = XGBoost.train(trainMatrix, boosterParams, numRound, new HashMap<>(), null, null);

            // Export the freshly trained booster to a temporary folder and reload it through loadModel so
            // that fit() returns a model built by the exact same code path as model import. This guarantees
            // a freshly-trained model scores identically to the same model once saved and reloaded, and
            // keeps a single load/scoring path (the LightGBM provider follows the same approach).
            exportDir = exportModel(booster);
            return loadModel(exportDir, schema);
        } catch (final XGBoostError | IOException | ModelLoadingException | NumberFormatException e) {
            throw new ModelTrainingException("Failed to train the XGBoost model.", e);
        } finally {
            if (booster != null) {
                booster.dispose();
            }
            if (trainMatrix != null) {
                trainMatrix.dispose();
            }
            deleteExportDir(exportDir);
        }
    }

    @Override
    public List<ParamValidationError> validateForFit(final Path pathToPersist,
                                                     final DatasetSchema schema,
                                                     final Map<String, String> params) {
        final ImmutableList.Builder<ParamValidationError> errorBuilder = ImmutableList.builder();

        errorBuilder.addAll(ValidationUtils.validateModelPathToTrain(pathToPersist));
        errorBuilder.addAll(ValidationUtils.checkParams(
                XgboostAlgorithms.XGBOOST_BINARY_CLASSIFIER.getAlgorithmDescriptor(), params));
        ValidationUtils.validateCategoricalSchema(schema).ifPresent(errorBuilder::add);
        if (schemaHasStringFields(schema)) {
            errorBuilder.add(new ParamValidationError(ERROR_MSG_SCHEMA_HAS_STRING_FIELDS));
        }
        if (getNumTargetClasses(schema).orElse(-1) != 2) {
            errorBuilder.add(new ParamValidationError(ERROR_MSG_NON_BINARY_TARGET));
        }

        return errorBuilder.build();
    }

    /**
     * Checks whether the schema contains any free-text (string) field. XGBoost consumes only numeric
     * input, and Pulse encodes categorical values as numeric indices, so numeric and categorical fields
     * are supported but {@link StringValueSchema} fields are not.
     *
     * @param schema The schema to inspect.
     * @return {@code true} if at least one field uses a {@link StringValueSchema}.
     */
    private static boolean schemaHasStringFields(final DatasetSchema schema) {
        return schema.getFieldSchemas().stream()
                .anyMatch(field -> field.getValueSchema() instanceof StringValueSchema);
    }

    /**
     * Returns the number of classes of the schema's target field, mirroring the LightGBM provider. The
     * result is empty when there is no target field or the target is not categorical, which the binary
     * validation treats the same as a non-binary target.
     *
     * @param schema The schema to inspect.
     * @return The number of target classes, or empty if there is no categorical target field.
     */
    private static Optional<Integer> getNumTargetClasses(final DatasetSchema schema) {
        return schema.getTargetFieldSchema()
                .map(FieldSchema::getValueSchema)
                .filter(valueSchema -> valueSchema instanceof CategoricalValueSchema)
                .map(valueSchema -> ((CategoricalValueSchema) valueSchema).getNominalValues().size());
    }

    /**
     * Materializes the whole dataset into an in-memory dense {@link DMatrix} with its label column set.
     *
     * @param dataset The training dataset.
     * @return The training {@link DMatrix}.
     * @throws XGBoostError           If the native matrix cannot be created.
     * @throws ModelTrainingException If the dataset is empty.
     */
    private static DMatrix buildTrainMatrix(final Dataset dataset) throws XGBoostError, ModelTrainingException {
        final DatasetSchema schema = dataset.getSchema();
        final int numFeatures = XgboostSchemaUtils.numFeatures(schema);
        // Supervised training requires a target; enforced by validateForFit via validateCategoricalSchema.
        final int targetIndex = schema.getTargetIndex().orElseThrow(
                () -> new IllegalStateException("Supervised training requires a schema with a target field."));

        final List<float[]> rows = new ArrayList<>();
        final List<Float> labels = new ArrayList<>();

        final Iterator<Instance> iterator = dataset.getInstances();
        while (iterator.hasNext()) {
            final Instance instance = iterator.next();
            labels.add((float) instance.getValue(targetIndex));
            rows.add(XgboostSchemaUtils.featureRow(instance, schema));
        }

        final int numRows = rows.size();
        if (numRows == 0) {
            throw new ModelTrainingException("Received an empty training dataset for XGBoost.");
        }

        final float[] flatFeatures = new float[numRows * numFeatures];
        final float[] labelArray = new float[numRows];
        for (int row = 0; row < numRows; row++) {
            System.arraycopy(rows.get(row), 0, flatFeatures, row * numFeatures, numFeatures);
            labelArray[row] = labels.get(row);
        }

        final DMatrix trainMatrix = new DMatrix(flatFeatures, numRows, numFeatures, MISSING_VALUE);
        trainMatrix.setLabel(labelArray);
        return trainMatrix;
    }

    /**
     * Translates the Pulse string parameters into the {@code Map<String, Object>} expected by
     * {@code xgboost4j}. The {@value XgboostDescriptorUtil#NUM_ROUND_PARAMETER_NAME} entry is excluded
     * because it is passed as the {@code nrounds} argument of {@link XGBoost#train}. A seed is derived
     * from the supplied {@link Random} when not explicitly provided, for reproducibility.
     *
     * @param params The Pulse model parameters.
     * @param random The source of randomness.
     * @return The XGBoost booster parameters.
     */
    static Map<String, Object> toBoosterParams(final Map<String, String> params, final Random random) {
        final Map<String, Object> boosterParams = new HashMap<>();

        params.forEach((name, value) -> {
            if (!XgboostDescriptorUtil.NUM_ROUND_PARAMETER_NAME.equals(name) && value != null && !value.isEmpty()) {
                boosterParams.put(name, value);
            }
        });

        boosterParams.putIfAbsent(XgboostDescriptorUtil.OBJECTIVE_PARAMETER_NAME, "binary:logistic");
        boosterParams.putIfAbsent(XgboostDescriptorUtil.SEED_PARAMETER_NAME, random.nextInt(Integer.MAX_VALUE));

        return boosterParams;
    }

    /**
     * Reads the number of boosting rounds from the parameters, falling back to a default.
     *
     * @param params The Pulse model parameters.
     * @return The number of boosting rounds.
     */
    static int numRoundOf(final Map<String, String> params) {
        final String numRound = params.get(XgboostDescriptorUtil.NUM_ROUND_PARAMETER_NAME);
        if (numRound == null || numRound.isEmpty()) {
            return DEFAULT_NUM_ROUND;
        }
        return Integer.parseInt(numRound.trim());
    }

    /**
     * Exports a trained booster following the Pulse model folder convention
     * ({@code <exportDir>/model/<file>}), so it can be reloaded through {@link #loadModel}.
     *
     * @param booster The trained booster.
     * @return The export directory root.
     * @throws IOException  If the export directories/files cannot be created.
     * @throws XGBoostError If the booster cannot be serialized.
     */
    private static Path exportModel(final Booster booster) throws IOException, XGBoostError {
        final Path exportDir = Files.createTempDirectory(EXPORT_DIR_PREFIX);
        final Path modelDir = Files.createDirectory(exportDir.resolve(LoadModelUtils.MODEL_FOLDER));
        booster.saveModel(modelDir.resolve(MODEL_BINARY_RESOURCE_FILE_NAME).toString());
        return exportDir;
    }

    /**
     * Best-effort recursive deletion of the temporary export directory created by {@link #exportModel}.
     * The directory only exists to hand the freshly trained model to {@link #loadModel}; once the model
     * is loaded into memory the files are no longer needed. Failures are logged and swallowed, mirroring
     * how the LightGBM provider cleans up its temporary model file.
     *
     * @param exportDir The temporary export directory, or {@code null} if it was never created.
     */
    static void deleteExportDir(final Path exportDir) {
        if (exportDir == null) {
            return;
        }
        try {
            MoreFiles.deleteRecursively(exportDir, RecursiveDeleteOption.ALLOW_INSECURE);
        } catch (final IOException e) {
            logger.warn("Could not clean up temporary export directory [{}]: {}", exportDir, e.getMessage());
        }
    }
}
