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

package com.feedzai.openml.h2o.server;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.h2o.H2OAlgorithm;
import com.feedzai.openml.h2o.H2OConverter;
import com.feedzai.openml.h2o.algos.H2OBayesUtils;
import com.feedzai.openml.h2o.algos.H2ODeepLearningUtils;
import com.feedzai.openml.h2o.algos.H2ODrfUtils;
import com.feedzai.openml.h2o.algos.H2OGbmUtils;
import com.feedzai.openml.h2o.algos.H2OGeneralizedLinearModelUtils;
import com.feedzai.openml.h2o.algos.H2OIsolationForestUtils;
import com.feedzai.openml.h2o.algos.H2OXgboostUtils;
import com.feedzai.openml.h2o.server.export.MojoExported;
import com.feedzai.openml.h2o.server.export.PojoExported;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.MachineLearningAlgorithmType;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;
import hex.Model;
import hex.deeplearning.DeepLearning;
import hex.glm.GLM;
import hex.naivebayes.NaiveBayes;
import hex.schemas.DRFV3;
import hex.schemas.DeepLearningV3;
import hex.schemas.GBMV3;
import hex.schemas.GLMV3;
import hex.schemas.IsolationForestV3;
import hex.schemas.NaiveBayesV3;
import hex.schemas.XGBoostV3;
import hex.tree.drf.DRF;
import hex.tree.gbm.GBM;
import hex.tree.isofor.IsolationForest;
import hex.tree.isofor.IsolationForestModel;
import hex.tree.isofor.IsolationForestModel.IsolationForestOutput;
import hex.tree.xgboost.XGBoost;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import water.ExtensionManager;
import water.H2O;
import water.Job;
import water.Key;
import water.fvec.Frame;
import water.fvec.NFSFileVec;
import water.parser.DefaultParserProviders;
import water.parser.ParseDataset;
import water.parser.ParseSetup;

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

/**
 * A class representing an H2O instance from which common operations (such as training models) can be requested.
 *
 * @since 0.1.0
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 * @param <M> The specific {@link Model}.
 */
public class H2OApp<M extends Model> {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OApp.class);

    /**
     * The singleton instance.
     * @since 0.1.0
     */
    private static volatile H2OApp instance;

    /**
     * The lock used to initialize the instance.
     * @since 0.1.0
     */
    private final static Object instanceLock = new Object();

    /**
     * Gets the {@link H2OApp} singleton instance.
     *
     * @return The instance.
     * @since 0.1.0
     */
    public static H2OApp getInstance() {
         if (instance == null) {
            synchronized (instanceLock) {
                if (instance == null)
                    instance = new H2OApp();
            }
        }
        return instance;
    }

    /**
     * Private constructor for singleton pattern.
     *
     * @implNote The instance of H2O is initialized in a thread of the current process, and once initialized is not
     * possible to shutdown or to initialize another instance. The only way to shutdown H2O is to call 'H2O.shutdown(0)'
     * but that will also stop the current process (it calls System.exit(int)). A workaround for this problem it to use
     * a singleton instance of H2O.
     * By default, when a H2O instance is created it retrieves the list of available network interfaces to search for
     * nodes with the same cluster name. Therefore the isolation of nodes should be guaratee, each cluster should have
     * an unique name and their addresses should be set to localhost. This way they wouldn't search for nodes in the
     * available network interfaces.
     *
     * @see <a href="http://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html#h2o-networking">H2O Networking</a>
     */
    private H2OApp() {
        String hostName = "";
        try {
            hostName = "_".concat(InetAddress.getLocalHost().getHostAddress());
        } catch (final UnknownHostException e) {
            logger.warn("An error was detected while getting the local IP address", e);
        }
        H2O.main(new String[] {"-name", String.format("h2o%s_%s", hostName, UUID.randomUUID()), "-ip", "localhost"});
        ExtensionManager.getInstance().registerRestApiExtensions();
        water.util.Log.setLogLevel("WARN", true);
    }

    /**
     * Trains a model for the given algorithm.
     *
     * @param algorithm   The {@link MLAlgorithmDescriptor algorithm} to train.
     * @param datasetPath The location of the dataset.
     * @param schema      The dataset schema.
     * @param params      The algorithm params.
     * @param randomSeed  The source of randomness.
     * @return The id of the trained model for the specified algorithm.
     * @throws ModelTrainingException If any problem occurs during the train.
     */
    public Model train(final MLAlgorithmDescriptor algorithm,
                       final Path datasetPath,
                       final DatasetSchema schema,
                       final Map<String, String> params,
                       final long randomSeed) throws ModelTrainingException {
        logger.info("Training {} algorithm from dataset in {}. TargetIdx: {}, params: {}", algorithm, datasetPath, schema.getTargetIndex(), params);
        final Frame dataset = parseDataSetFile(datasetPath, schema);

        if (algorithm.getAlgorithmType() == MachineLearningAlgorithmType.ANOMALY_DETECTION) {
            return trainAnomalyDetection(algorithm, dataset, params, randomSeed, schema);
        }


        final Optional<FieldSchema> targetFieldSchema = schema.getTargetFieldSchema();
        if (!targetFieldSchema.isPresent()) {
            throw new ModelTrainingException("In order to train a supervised model, please provide a schema with a target field.");
        }
        return trainSupervised(algorithm, dataset, targetFieldSchema.get(), params, randomSeed);
    }

    /**
     * Exports the model to the specified path.
     *
     * @param model     The model to export.
     * @param exportDir The directory where to save the exported model.
     * @return The path to the exported model.
     * @throws IOException If cannot write model to disk.
     * @throws ModelTrainingException If any problem occurs saving the model into disk.
     */
    public Path export(final Model model,
                       final Path exportDir) throws IOException, ModelTrainingException {
        logger.info("Exporting model {} to {}", model._output._job._result.toString(), exportDir.toAbsolutePath().toString());
        if (model.haveMojo()) {
            if (model instanceof IsolationForestModel && ((IsolationForestModel) model)._output._min_path_length == Long.MAX_VALUE) {
                ((IsolationForestModel) model)._output._min_path_length = Integer.MAX_VALUE;
            }
            new MojoExported().save(exportDir, model);
        } else {
            new PojoExported().save(exportDir, model);
        }

        return exportDir;
    }

    private Model trainAnomalyDetection(final MLAlgorithmDescriptor algorithmDescriptor,
                                        final Frame trainingFrame,
                                        final Map<String, String> params,
                                        final long randomSeed, final DatasetSchema datasetSchema) throws ModelTrainingException {
        final Job<M> modelJob;

        final H2OAlgorithm h2OAlgorithm = getH2OAlgorithm(algorithmDescriptor);

        switch (h2OAlgorithm) {
            case ISOLATION_FOREST:
                final IsolationForestV3.IsolationForestParametersV3 isolationForestParams = new H2OIsolationForestUtils().parseParams(trainingFrame, params, randomSeed, datasetSchema);

                modelJob = train(() -> new IsolationForest(isolationForestParams.createAndFillImpl()).trainModel())
                        .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));
                break;

            default:
                logger.error("Training not supported for algorithm {}", algorithmDescriptor.getAlgorithmName());
                throw new IllegalArgumentException("Unsupported anomaly detection algorithm: " + algorithmDescriptor);
        }

        final M model = waitForModelTrained(modelJob);
        return model;
    }

    private H2OAlgorithm getH2OAlgorithm(final MLAlgorithmDescriptor algorithmDescriptor) {
        return MLAlgorithmEnum.getByName(H2OAlgorithm.values(), algorithmDescriptor.getAlgorithmName())
                .orElseThrow(() -> new IllegalArgumentException("Unknown algorithm: " + algorithmDescriptor));
    }

    /**
     * Trains the specified supervised algorithm in the H2O platform.
     *
     * @param algorithmDescriptor The algorithm to train.
     * @param trainingFrame       The {@link Frame dataset} to use during training.
     * @param targetField         The target field.
     * @param params              The algorithm params.
     * @param randomSeed          The source of randomness.
     * @return The {@link Model model}.
     * @throws ModelTrainingException If any problem occurred during the train of the model,
     *                                such as problems connecting to the H2O server.
     */
    private Model trainSupervised(final MLAlgorithmDescriptor algorithmDescriptor,
                                  final Frame trainingFrame,
                                  final FieldSchema targetField,
                                  final Map<String, String> params,
                                  final long randomSeed) throws ModelTrainingException {
        final Job<M> modelJob;

        final H2OAlgorithm h2OAlgorithm = getH2OAlgorithm(algorithmDescriptor);

        final int targetIndex = targetField.getFieldIndex();

        switch (h2OAlgorithm) {
            case DISTRIBUTED_RANDOM_FOREST:
                final DRFV3.DRFParametersV3 drfParams = new H2ODrfUtils().parseParams(trainingFrame, targetIndex, params, randomSeed);

                modelJob = train(() -> new DRF(drfParams.createAndFillImpl()).trainModel())
                        .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));
                break;

            case XG_BOOST:
                final XGBoostV3.XGBoostParametersV3 xgboostParams = new H2OXgboostUtils().parseParams(trainingFrame, targetIndex, params, randomSeed);

                modelJob = train(() -> new XGBoost(xgboostParams.createAndFillImpl()).trainModel())
                        .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));
                break;

            case DEEP_LEARNING:
                final DeepLearningV3.DeepLearningParametersV3 deepLearningParams = new H2ODeepLearningUtils().parseParams(trainingFrame, targetIndex, params, randomSeed);

                modelJob = train(() -> new DeepLearning(deepLearningParams.createAndFillImpl()).trainModel())
                        .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));
                break;

            case GRADIENT_BOOSTING_MACHINE:
                final GBMV3.GBMParametersV3 gbmParams = new H2OGbmUtils().parseParams(trainingFrame, targetIndex, params, randomSeed);

                modelJob = train(() -> new GBM(gbmParams.createAndFillImpl()).trainModel())
                        .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));
                break;

            case NAIVE_BAYES_CLASSIFIER:
                final NaiveBayesV3.NaiveBayesParametersV3 naiveBayesParams = new H2OBayesUtils().parseParams(trainingFrame, targetIndex, params, randomSeed);
                modelJob = train(() -> new NaiveBayes(naiveBayesParams.createAndFillImpl()).trainModel())
                        .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));
                break;

            case GENERALIZED_LINEAR_MODEL:
                final GLMV3.GLMParametersV3 glmParams = new H2OGeneralizedLinearModelUtils(targetField).parseParams(trainingFrame, targetIndex, params, randomSeed);

                modelJob = train(() -> new GLM(glmParams.createAndFillImpl()).trainModel())
                        .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));
                break;

            default:
                logger.error("Training not supported for algorithm {}", algorithmDescriptor.getAlgorithmName());
                throw new IllegalArgumentException("Unsupported supervised algorithm: " + algorithmDescriptor);
        }

        return waitForModelTrained(modelJob);
    }

    /**
     * Creates a standard {@link ModelTrainingException} with a common message template.
     *
     * @param algorithm The {@link MLAlgorithmDescriptor} request to train.
     * @return The {@link ModelTrainingException}
     */
    private ModelTrainingException createModelTrainingException(final MLAlgorithmDescriptor algorithm) {
        return new ModelTrainingException(String.format("Error training model (%s) on h2o", algorithm));
    }

    /**
     * Trains a model and wraps the possible returned errors into a {@link ModelTrainingException}.
     *
     * @param trainer The entity that knows how to train a specific model.
     * @return The {@link Job job of the model}.
     * @throws ModelTrainingException If any problem occurs during the training.
     */
    private Optional<Job<M>> train(final Trainer trainer) throws ModelTrainingException {
        try {
            return Optional.ofNullable(trainer.train());
        } catch (final IOException e) {
            logger.warn("Error training algorithm", e);
            throw new ModelTrainingException(e);
        }
    }

    /**
     * Waits for a model to finish its training.
     *
     * @param modelJob The model to check for train completeness.
     * @return The {@link Model} trained.
     */
    private M waitForModelTrained(final Job<M> modelJob) {
        logger.info("Waiting for model (job {}) to finish train", modelJob._key);
        return modelJob.get();
    }

    /**
     * Finds and parses a Dataset in a CSV file.
     *
     * @param dataSetFile DataSet filename.
     * @param schema The {@link DatasetSchema}-
     * @return Frame or NPE.
     *
     * @implNote This method needs to be synchronized. While we can't really pinpoint the exact reason, the experience
     * with tests training multiple models concurrently shows that this method is not thread safe.
     */
    private synchronized Frame parseDataSetFile(final Path dataSetFile, final DatasetSchema schema) {
        final NFSFileVec nfs = NFSFileVec.make(dataSetFile.toFile());

        final ParseSetup userSetup = new ParseSetup();
        userSetup.setParseType(DefaultParserProviders.GUESS_INFO);

        final Key[] keys = new Key[]{nfs._key};

        final ParseSetup setup = ParseSetup.guessSetup(keys, userSetup);
        setup.setColumnNames(H2OConverter.convertColumnNames(schema.getFieldSchemas()));
        setup.setColumnTypes(ParseSetup.strToColumnTypes(H2OConverter.convertColumnTypes(schema.getFieldSchemas())));
        setup.setNumberColumns(schema.getFieldSchemas().size());
        setup.setSeparator((byte) ','); // TODO PULSEDEV-23507
        setup.setDomains(H2OConverter.convertDomains(schema.getFieldSchemas())); // TODO PULSEDEV-23525 there's still some error with cross-validation)

        return ParseDataset.parse(Key.make(), keys, true, setup);
    }


}
