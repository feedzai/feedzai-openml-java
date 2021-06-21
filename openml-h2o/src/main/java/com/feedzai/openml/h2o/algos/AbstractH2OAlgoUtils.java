/*
 * Copyright 2019 Feedzai
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

package com.feedzai.openml.h2o.algos;

import com.feedzai.openml.data.schema.DatasetSchema;
import hex.ModelBuilder;
import org.apache.commons.lang3.StringUtils;
import water.Job;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;

import java.util.Map;
import java.util.Optional;

/**
 * Abstract class to parse H2O algorithm params, validate them and train the model.
 *
 * @param <T> The concrete type of {@link ModelParametersSchemaV3 algorithm params}.
 * @param <M> The concrete type of {@link ModelBuilder H20 Model Builder}.
 * @since 0.1.0 //todo: validate if this should stay or be changed since this class went under serious changes (including name and generic param)
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 * @author Antonio Silva (antonio.silva@feedzai.com)
 */
public abstract class AbstractH2OAlgoUtils<T extends ModelParametersSchemaV3, M extends ModelBuilder> {

    /**
     * Cleans a parameter value.
     *
     * @param param The raw parameter value.
     * @return An optional containing the processed param value, or an empty optional if the raw value is not meaningful.
     */
    Optional<String> cleanParam(final String param) {
        if (StringUtils.isBlank(param)) {
            return Optional.empty();
        } else {
            return Optional.of(StringUtils.trim(param));
        }
    }

    /**
     * Parses algorithm specific parameters from the raw params.
     *
     * @param h2oParams The {@link ModelParametersSchemaV3} to be filled.
     * @param params The raw training params.
     * @param randomSeed The source of randomness.
     * @return The modified version of the given {@code h2oParams}.
     */
    protected abstract T parseSpecificParams(final T h2oParams, final Map<String, String> params, final long randomSeed);

    /**
     * Returns an empty representation of the algorithm specific parameters.
     *
     * @return An empty representation of the algorithm specific parameters.
     */
    protected abstract T getEmptyParams();

    /**
     * Template method to parse H2O algorithm params.
     *
     * @param trainingFrame The dataset to be used.
     * @param params        The raw training params.
     * @param randomSeed    The source of randomness.
     * @param datasetSchema The dataset schema.
     * @return The modified version of the given {@code h2oParams}.
     */
    protected final T parseParams(final Frame trainingFrame,
                               final Map<String, String> params,
                               final long randomSeed,
                               final DatasetSchema datasetSchema) {
        final T baseParams = commonParams(trainingFrame, datasetSchema);
        return parseSpecificParams(baseParams, params, randomSeed);
    }

    /**
     * Auxiliary method to setup the common training params to all algorithms/models.
     *
     * @param trainingFrame The dataset to be used.
     * @param datasetSchema The dataset schema.
     * @return A modified version of the provided params object.
     */
    protected abstract T commonParams(Frame trainingFrame, DatasetSchema datasetSchema);

    /**
     * Abstract method that creates and returns the Model that this class refers to.
     * @param h2oParams concrete implementation of {@link ModelParametersSchemaV3 algorithm params}.
     * @return concrete implementation of {@link ModelBuilder H20 Model Builder} created from the h2oParams.
     *
     * @since @@@feedzai.next.release@@@
     */
    public abstract M getModel(T h2oParams);

    /**
     * Validates the h20 model parameters by building a model and validating it.
     * @param paramsToValidate H20 parameters to validate.
     * @param randomSeed       The source of randomness.
     *
     * @since @@@feedzai.next.release@@@
     */
    public void validateParams(Map<String, String> paramsToValidate, final long randomSeed){
        final T baseParams = getEmptyParams();
        final T updatedParams = parseSpecificParams(baseParams, paramsToValidate, randomSeed);
        M modelBuilder = getModel(updatedParams);
        validateModel(modelBuilder);
    }

    /**
     * Trains the H20 model.
     * @param trainingFrame The dataset to be used.
     * @param params        The raw training params.
     * @param randomSeed    The source of randomness.
     * @param schema        The dataset schema.
     * @return Job resulting from he model training.
     *
     * @since @@@feedzai.next.release@@@
     */
    public Job train(final Frame trainingFrame,
                     final Map<String, String> params,
                     final long randomSeed,
                     final DatasetSchema schema){
        M model = getModel(trainingFrame, params, randomSeed, schema);
        return model.trainModel();

    }

    /**
     * Method that returns creates the model in a training scenario.
     * @param trainingFrame The dataset to be used.
     * @param params        The raw training params.
     * @param randomSeed    The source of randomness.
     * @param schema        The dataset schema.
     * @return
     * @since @@@feedzai.next.release@@@
     */
    private M getModel(final Frame trainingFrame,
                       final Map<String, String> params,
                       final long randomSeed,
                       final DatasetSchema schema) {
        final T parsedParams = parseParams(trainingFrame, params, randomSeed, schema);
        return getModel(parsedParams);
    }

    /**
     *
     * @param model
     *
     * @since @@@feedzai.next.release@@@
     */
    private void validateModel(final M model) {
        if (model.error_count() > 0) {
            //TODO: create a custom exception to throw ?
            throw new IllegalArgumentException("Model has errors: " + model.validationErrors());
        }
    }
}
