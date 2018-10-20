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
import com.feedzai.openml.h2o.server.H2OApp;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.google.common.base.MoreObjects;
import hex.Model;
import hex.VarImp;
import hex.tree.gbm.GBMModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

/**
 * A class that provides logic to calculate the importance for each feature in a dataset, by using H2O's features.
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 0.1.0
 */
public class H2OFeatureImportanceCreator {

    /**
     * Logger for {@link H2OFeatureImportanceCreator}.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OFeatureImportanceCreator.class);

    /**
     * The h2OApp used by this model creator.
     */
    private final H2OApp h2OApp;

    /**
     * Default constructor. Uses {@link H2OApp#getInstance()}.
     */
    public H2OFeatureImportanceCreator() {
        this(H2OApp.getInstance());
    }

    /**
     * Alternative constructor for this class. Allows definition of the actual {@link H2OApp} instance used.
     *
     * @param h2OApp The {@link H2OApp} used for feature importance calculation.
     */
    public H2OFeatureImportanceCreator(final H2OApp h2OApp) {
        this.h2OApp = h2OApp;
    }

    /**
     * Calculates feature importance for the provided {@link Dataset dataset}. This method trains a Gradient Boost Machine model and fetches the feature importance results
     * from the GBM {@link Model}.
     *
     * <p>
     *     Note that the algorithm bound to this object has no effect on the outcome of this method, and it will always use GBM.
     * </p>
     *
     * @param dataset The dataset used to calculate feature importance.
     * @param random  A random generator used to provide a random seed to the model training operation.
     * @param params  The parameters for the GBM algorithm.
     * @return A {@link VarImp} object with the feature importance results.
     *
     * @throws ModelTrainingException if the feature importance task fails while training the GBM model.
     *
     * @see H2OAlgorithm#GRADIENT_BOOSTING_MACHINE
     * @see H2OApp#train(MLAlgorithmDescriptor, Path, DatasetSchema, Map, long)
     *
     * @since 0.1.0
     */
    public VarImp calculateFeatureImportance(final Dataset dataset,
                                             final Random random,
                                             final Map<String, String> params) throws ModelTrainingException {
        final H2OAlgorithm gbm = H2OAlgorithm.GRADIENT_BOOSTING_MACHINE;

        try {
            final Path datasetPath = H2OUtils.writeDatasetToDisk(dataset);
            final Model model = this.h2OApp.train(gbm.getAlgorithmDescriptor(), datasetPath, dataset.getSchema(), params, random.nextLong());
            // safe cast as long as using H2OAlgorithm.GRADIENT_BOOSTING_MACHINE
            return ((GBMModel) model)._output._varimp;
        } catch (final IOException e) {
            final String errorMessage = "An error occurred while calculating feature importance.";
            logger.error(errorMessage, e);
            throw new ModelTrainingException(errorMessage, e);
        }
    }

    @Override
    public String toString() {
        return MoreObjects.toStringHelper(this)
                .add("h2OApp", this.h2OApp)
                .toString();
    }
}
