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

package com.feedzai.openml.datarobot;

import com.datarobot.prediction.Predictor;
import com.datarobot.prediction.Row;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.AbstractValueSchema;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.model.ClassificationMLModel;
import com.feedzai.openml.model.MachineLearningModel;
import com.feedzai.openml.util.data.encoding.EncodingHelper;
import com.google.common.base.Preconditions;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Classification object used to represent a binary {@link MachineLearningModel model} generated in DataRobot.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public class ClassificationBinaryDataRobotModel implements ClassificationMLModel {

    /**
     * Logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(ClassificationBinaryDataRobotModel.class);

    /**
     * The predictor for the binary model generated in DataRobot.
     */
    private final Predictor predictor;

    /**
     * The path from where the model was initially loaded.
     */
    private final Path modelPath;

    /**
     * The {@link DatasetSchema} the model uses.
     */
    private final DatasetSchema schema;

    /**
     * A {@link URLClassLoader} that needs to be closed upon {@link #close()}.
     */
    private final URLClassLoader urlClassLoader;

    /**
     * True if the first nominal value declared on the schema was used as target value to train the model on DataRobot,
     * false otherwise.
     */
    private final boolean firstNominalValueUsedToTrain;

    /**
     * {@link Map} with the names and indexes of each {@link FieldSchema} declared in the {@link DatasetSchema}.
     */
    private final Map<String, Integer> mapFieldNameIndex;

    /**
     * {@link Map} with the indexes and nominal values of each {@link CategoricalValueSchema} declared in the
     * {@link DatasetSchema}.
     */
    private final Map<Integer, String[]> mapCatIndexNominalValue;

    /**
     * Constructor for a {@link ClassificationBinaryDataRobotModel}.
     *
     * @param predictor                    Predictor for the binary model generated in DataRobot.
     * @param firstNominalValueUsedToTrain True if the first nominal value declared on the schema was used as target
     *                                     value to train the model on DataRobot, false otherwise.
     * @param modelPath                    The path from where the model was initially loaded.
     * @param schema                       The {@link DatasetSchema} the model uses.
     * @param urlClassLoader               A {@link URLClassLoader} that needs to be closed upon {@link #close()}.
     */
    ClassificationBinaryDataRobotModel(final Predictor predictor,
                                       final boolean firstNominalValueUsedToTrain,
                                       final Path modelPath,
                                       final DatasetSchema schema,
                                       final URLClassLoader urlClassLoader) {
        this.predictor = Preconditions.checkNotNull(predictor, "predictor cannot be null");
        this.firstNominalValueUsedToTrain = firstNominalValueUsedToTrain;
        this.modelPath = Preconditions.checkNotNull(modelPath, "path of the model cannot be null");
        this.schema = Preconditions.checkNotNull(schema, "dataset schema cannot be null");
        this.urlClassLoader = Preconditions.checkNotNull(urlClassLoader, "the urlClassLoader cannot be null");
        this.mapFieldNameIndex = createMapOfFieldNamesAndIndexes();
        this.mapCatIndexNominalValue = createMapOfCatFieldsAndDecodedValues();
    }

    /**
     * Responsible for the creation of {@link #mapFieldNameIndex}.
     *
     * @return {@link #mapFieldNameIndex}.
     */
    private Map<String, Integer> createMapOfFieldNamesAndIndexes() {
        return this.schema.getFieldSchemas()
                .stream()
                .collect(Collectors.toMap(FieldSchema::getFieldName, FieldSchema::getFieldIndex));
    }

    /**
     * Responsible for the creation of {@link #mapCatIndexNominalValue}.
     *
     * @return {@link #mapCatIndexNominalValue}.
     */
    private Map<Integer, String[]> createMapOfCatFieldsAndDecodedValues() {
        return this.schema.getFieldSchemas()
                .stream()
                .filter(field -> field.getValueSchema() instanceof CategoricalValueSchema)
                .collect(
                        Collectors.toMap(
                                FieldSchema::getFieldIndex,
                                field -> {
                                    final CategoricalValueSchema valueSchema = (CategoricalValueSchema) field.getValueSchema();
                                    final int numOfCategories = valueSchema.getNominalValues().size();
                                    final String[] decodedCategories = new String[numOfCategories];
                                    for (int index = 0; index < numOfCategories; index++) {
                                        decodedCategories[index] = EncodingHelper.decodeDoubleToCategory(index, valueSchema);
                                    }
                                    return decodedCategories;
                                }
                        )
                );
    }

    @Override
    public double[] getClassDistribution(final Instance instance) {
        final double score = predictProbOfFirstTargetValue(instance);
        return new double[]{score, 1.0 - score};
    }

    /**
     * {@inheritDoc}
     *
     * @implNote This method assumes that the target field only has two possible values and that the value returned by
     * {@link #predictProbOfFirstTargetValue(Instance)} is a probability of the positive class. In other other this means that when
     * the {@code score} is rounded it will return 0 or 1.
     */
    @Override
    public int classify(final Instance instance) {
        final double score = predictProbOfFirstTargetValue(instance);
        return (int) Math.round(score);
    }

    @Override
    public boolean save(final Path dir, final String name) {
        try {
            FileUtils.copyDirectory(this.modelPath.toFile(), dir.toFile());
            return true;
        } catch (final IOException e) {
            final String msg = String.format("Error saving model %s to %s", name, dir);
            logger.error(msg, e);
            throw new RuntimeException(msg, e);
        }
    }

    @Override
    public DatasetSchema getSchema() {
        return this.schema;
    }

    @Override
    public void close() throws IOException {
        this.urlClassLoader.close();
    }

    /**
     * Predicts the probability for the value stored in the first nominal value of the
     * {@link DatasetSchema#getTargetFieldSchema() target field}.
     *
     * @param instance The {@link Instance} to be classified.
     * @return The probability of the first target value.
     */
    private double predictProbOfFirstTargetValue(final Instance instance) {
        try {
            final Row convertedInstance = convertInstanceToRowData(instance);
            double scoreOfValueAtFirstIndex = this.predictor.score(convertedInstance);
            if (!this.firstNominalValueUsedToTrain) {
                scoreOfValueAtFirstIndex = 1.0 - scoreOfValueAtFirstIndex;
            }
            return scoreOfValueAtFirstIndex;
        } catch (final Exception exception) {
            final String errorMsg = String.format(
                    "The model failed to classify the event [%s]!",
                    convertInstanceToString(instance)
            );
            logger.error(errorMsg);
            throw new RuntimeException(errorMsg, exception);
        }
    }

    /**
     * Converts the data type of an event in order to be classified by the DataRobot model.
     * It converts from {@link Instance} to {@link Row}.
     *
     * @param instance The {@link Instance} to be classified.
     * @return The {@link Row} to be classified.
     */
    private Row convertInstanceToRowData(final Instance instance) {
        final Row row = new Row();

        final String[] doublePredictors = this.predictor.get_double_predictors();
        final int numOfNumericFields = doublePredictors.length;
        row.d = new double[numOfNumericFields];
        for (int i = 0; i < numOfNumericFields; i++) {
            row.d[i] = instance.getValue(this.mapFieldNameIndex.get(doublePredictors[i]));
        }

        final String[] stringPredictors = this.predictor.get_string_predictors();
        final int numOfStringFields = stringPredictors.length;
        row.s = new String[numOfStringFields];
        for (int i = 0; i < numOfStringFields; i++) {
            final int index = this.mapFieldNameIndex.get(stringPredictors[i]);
            final AbstractValueSchema valueSchema = this.schema.getFieldSchemas().get(index).getValueSchema();

            if (valueSchema instanceof StringValueSchema) {
                row.s[i] = instance.getStringValue(index);
            } else if (valueSchema instanceof CategoricalValueSchema) {
                row.s[i] = this.mapCatIndexNominalValue.get(index)[(int) instance.getValue(index)];
            } else {
                row.s[i] = String.valueOf(instance.getValue(index));
            }
        }
        return row;
    }

    /**
     * Converts a {@link Instance} to a string that contains the values for each {@link FieldSchema field}.
     *
     * @param instance The {@link Instance} to be converted.
     * @return A string with the values of the {@link Instance}.
     */
    private String convertInstanceToString(final Instance instance) {
        return getSchema().getFieldSchemas()
                .stream()
                .map(fieldSchema -> {
                    if (fieldSchema.getValueSchema() instanceof StringValueSchema) {
                        return instance.getStringValue(fieldSchema.getFieldIndex());
                    }
                    return String.valueOf(instance.getValue(fieldSchema.getFieldIndex()));
                })
                .collect(Collectors.joining( "," ));
    }
}
