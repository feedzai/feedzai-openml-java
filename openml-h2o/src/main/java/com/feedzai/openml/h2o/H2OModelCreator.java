/*
 * Copyright 2018 Feedzai
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

package com.feedzai.openml.h2o;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.h2o.server.H2OApp;
import com.feedzai.openml.h2o.server.export.MojoExported;
import com.feedzai.openml.h2o.server.export.PojoExported;
import com.feedzai.openml.model.MachineLearningModel;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.feedzai.openml.provider.model.MachineLearningModelLoader;
import com.feedzai.openml.provider.model.MachineLearningModelTrainer;
import com.feedzai.openml.java.utils.JavaFileUtils;
import com.feedzai.openml.util.load.LoadModelUtils;
import com.feedzai.openml.util.load.LoadSchemaUtils;
import com.feedzai.openml.util.validate.ClassificationValidationUtils;
import com.feedzai.openml.util.validate.ValidationUtils;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import hex.Model;
import hex.genmodel.GenModel;
import hex.genmodel.MojoModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Implementation of the {@link MachineLearningModelLoader}.
 * <p>
 * This class is responsible for the initialization of a {@link MachineLearningModel} that was generated in H2O.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public class H2OModelCreator implements MachineLearningModelTrainer<ClassificationH2OModel> {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OModelCreator.class);

    /**
     * The {@link MLAlgorithmDescriptor} of the algorithm this creator refers to.
     */
    private final MLAlgorithmDescriptor algorithm;

    /**
     * The h2OApp used by this model creator.
     */
    private final H2OApp h2OApp;

    /**
     * Creates an instance for the specified {@link MLAlgorithmDescriptor}.
     *
     * @param algorithm The {@link MLAlgorithmDescriptor} that describes the algorithm to be instantiated.
     */
    public H2OModelCreator(final MLAlgorithmDescriptor algorithm) {
        this.algorithm = algorithm;
        this.h2OApp = H2OApp.getInstance();
    }

    /**
     * {@inheritDoc}
     *
     * This provider assumes the {@code modelPath} is a directory.
     */
    @Override
    public ClassificationH2OModel loadModel(final Path modelPath,
                                            final DatasetSchema schema) throws ModelLoadingException {

        logger.info("Trying to load a model in path [{}]...", modelPath);
        ClassificationValidationUtils.validateParamsModelToLoad(this, modelPath, schema, ImmutableMap.of());

        final GenModel genModel;
        final Closeable closeable;
        final String modelFilePath = LoadModelUtils.getModelFilePath(modelPath).toAbsolutePath().toString();

        final String fileExtension = com.google.common.io.Files.getFileExtension(modelFilePath);
        if (isPojo(fileExtension)) {
            final URLClassLoader urlClassLoader = JavaFileUtils.getUrlClassLoader(
                    modelFilePath,
                    ClassificationH2OModel.class.getClassLoader()
            );
            genModel = (GenModel) JavaFileUtils.createNewInstanceFromClassLoader(
                    modelFilePath,
                    "%s",
                    urlClassLoader
            );
            closeable = urlClassLoader;
        } else if (isMojo(fileExtension)) {
            genModel = importModelFromMOJO(modelFilePath);
            closeable = () -> { };
        } else {
            logger.error("Extension of the file [{}] not recognized for a H2O model.", modelFilePath);
            throw new ModelLoadingException(
                    String.format(
                            "Extension of the file [%s] not recognized for a H2O model. Supported extensions: %s, %s",
                            modelFilePath,
                            PojoExported.POJO_EXTENSION,
                            MojoExported.MOJO_EXTENSION
                    )
            );
        }

        if (!genModel.isSupervised()) {
            final String errorMsg = String.format("The model stored in [%s] is not supervised.", modelFilePath);
            logger.error(errorMsg);
            throw new ModelLoadingException(errorMsg);
        }

        final ClassificationH2OModel resultingModel = new ClassificationH2OModel(genModel, modelPath, schema, closeable);
        ClassificationValidationUtils.validateClassificationModel(schema, resultingModel);

        logger.info("Model loaded successfully.");
        return resultingModel;
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
        errorBuilder.addAll(ValidationUtils.checkNoFieldsOfType(schema, StringValueSchema.class));

        ValidationUtils.validateCategoricalSchema(schema).ifPresent(errorBuilder::add);

        return errorBuilder.build();
    }

    @Override
    public ClassificationH2OModel fit(final Dataset dataset,
                                      final Random random,
                                      final Map<String, String> params) throws ModelTrainingException {
        try {

            final Path datasetPath = H2OUtils.writeDatasetToDisk(dataset);
            final Model model = this.h2OApp.train(this.algorithm, datasetPath, dataset.getSchema(), params, random.nextLong());

            final Path exportPath = Files.createTempDirectory(H2OUtils.FEEDZAI_H2O_PREFIX + model._output._job._result.toString());
            final Path modelPath = Files.createDirectory(exportPath.resolve(LoadModelUtils.MODEL_FOLDER));
            this.h2OApp.export(model, modelPath);

            return loadModel(exportPath, dataset.getSchema());
        } catch (final ModelLoadingException e) {
            logger.error("Error loading trained and exported model", e);
            throw new ModelTrainingException("Error loading trained and exported model", e);
        } catch (final IOException e) {
            logger.error("Error training model.", e);
            throw new ModelTrainingException("Error training model", e);
        }
    }

    @Override
    public List<ParamValidationError> validateForFit(final Path pathToPersist,
                                                     final DatasetSchema schema,
                                                     final Map<String, String> params) {

        final ImmutableList.Builder<ParamValidationError> errorBuilder = ImmutableList.builder();

        errorBuilder.addAll(ValidationUtils.validateModelPathToTrain(pathToPersist));
        errorBuilder.addAll(ValidationUtils.checkParams(this.algorithm, params));
        ValidationUtils.validateCategoricalSchema(schema).ifPresent(errorBuilder::add);

        return errorBuilder.build();
    }


    /**
     * Checks if the extension of the file is compatible with MOJO.
     * This method assumes that the extension of a file with a MOJO model is equals to {#MOJO_EXTENSION}.
     *
     * @param fileExtension Extension of the file with a model.
     * @return True if the extension is compatible with MOJO, false otherwise.
     */
    private boolean isMojo(final String fileExtension) {
        return MojoExported.MOJO_EXTENSION.equals(fileExtension);
    }

    /**
     * Checks if the extension of the file is compatible with POJO.
     * This method assumes that the extension of a file with a POJO model is equals to {#JAR_EXTENSION}.
     *
     * @param fileExtension Extension of the file with a model.
     * @return True if the extension is compatible with POJO, false otherwise.
     */
    private boolean isPojo(final String fileExtension) {
        return PojoExported.POJO_EXTENSION.equals(fileExtension);
    }

    /**
     * Imports a model generated in H2O from a MOJO (Model Object, Optimized).
     *
     * @param modelPath Path of the binary with the model.
     * @return An instance of the generated model.
     * @throws ModelLoadingException If anything goes wrong.
     */
    private GenModel importModelFromMOJO(final String modelPath) throws ModelLoadingException {
        try {
            return MojoModel.load(modelPath);
        } catch (final IOException e) {
            logger.error("Could not load the model [{}].", modelPath, e);
            throw new ModelLoadingException(
                    String.format("An error was found during the import of the model [%s]", modelPath),
                    e
            );
        }
    }

    @Override
    public String toString() {
        return MoreObjects.toStringHelper(this)
                .add("algorithm", this.algorithm)
                .add("h2OApp", this.h2OApp)
                .toString();
    }
}
