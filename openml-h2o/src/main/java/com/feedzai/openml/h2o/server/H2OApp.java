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
import com.feedzai.openml.h2o.H2OConverter;
import com.feedzai.openml.h2o.algos.AbstractH2OAlgoUtils;
import com.feedzai.openml.h2o.algos.H2OAlgoUtilsFactory;
import com.feedzai.openml.h2o.server.export.MojoExported;
import com.feedzai.openml.h2o.server.export.PojoExported;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.descriptor.MachineLearningAlgorithmType;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelTrainingException;

import hex.Model;
import hex.tree.isofor.IsolationForestModel;
import java.util.List;
import java.util.Random;
import org.apache.commons.io.FileUtils;
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

        if (algorithm.getAlgorithmType() != MachineLearningAlgorithmType.ANOMALY_DETECTION && !schema.getTargetFieldSchema().isPresent()) {
            throw new ModelTrainingException("In order to train a supervised model, please provide a schema with a target field.");
        }

        return train(algorithm, dataset, schema, params, randomSeed);
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
            // this workaround prevents an H2O bug where this parameter deserialization fails when it is deserialized as an integer value.
            if (model instanceof IsolationForestModel && ((IsolationForestModel) model)._output._min_path_length == Long.MAX_VALUE) {
                ((IsolationForestModel) model)._output._min_path_length = Integer.MAX_VALUE;
            }
            new MojoExported().save(exportDir, model);
        } else {
            new PojoExported().save(exportDir, model);
        }

        return exportDir;
    }

    /**
     * Trains the specified algorithm in the H2O platform.
     *
     * @param algorithmDescriptor The algorithm to train.
     * @param trainingFrame       The {@link Frame dataset} to use during training.
     * @param schema              The schema for the model to be trained.
     * @param params              The algorithm params.
     * @param randomSeed          The source of randomness.
     * @return The {@link Model model}.
     * @throws ModelTrainingException If any problem occurred during the train of the model,
     *                                such as problems connecting to the H2O server.
     */
    private Model train(final MLAlgorithmDescriptor algorithmDescriptor,
                        final Frame trainingFrame,
                        final DatasetSchema schema,
                        final Map<String, String> params,
                        final long randomSeed) throws ModelTrainingException {

        AbstractH2OAlgoUtils h2OAlgoUtils = H2OAlgoUtilsFactory.getH2OAlgoUtils(algorithmDescriptor, schema);

        final Job<M> modelJob = train(() -> h2OAlgoUtils.train(trainingFrame, params, randomSeed, schema))
                .orElseThrow(() -> createModelTrainingException(algorithmDescriptor));

        return waitForModelTrained(modelJob);
    }

    /**
     * Validates the specified algorithm in the H2O platform.
     * @param algorithmDescriptor The algorithm to train.
     * @param schema              The schema for the model to be trained.
     * @param params              The algorithm params.
     * @return list of {@link ParamValidationError} validation errors
     *
     * @since @@@feedzai.next.release@@@
     */
    public List<ParamValidationError> validate(final MLAlgorithmDescriptor algorithmDescriptor,
                                               final DatasetSchema schema,
                                               final Map<String, String> params) {

        AbstractH2OAlgoUtils h2OAlgoUtils = H2OAlgoUtilsFactory.getH2OAlgoUtils(algorithmDescriptor, schema);

        final long randomSeed = new Random().nextLong();
        return h2OAlgoUtils.validateParams(params, randomSeed);

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
     * @throws ModelTrainingException if the dataset file is empty
     *
     * @implNote This method needs to be synchronized. While we can't really pinpoint the exact reason, the experience
     * with tests training multiple models concurrently shows that this method is not thread safe.
     */
    private synchronized Frame parseDataSetFile(final Path dataSetFile,
                                                final DatasetSchema schema) throws ModelTrainingException{
        if (FileUtils.sizeOf(dataSetFile.toFile()) == 0) {
            logger.info("The file: {} is empty. Cannot generate the model.", dataSetFile);
            throw new ModelTrainingException(
                    String.format("In order to generate the model the dataset cannot be empty. File: %s is empty.", dataSetFile));
        }

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
