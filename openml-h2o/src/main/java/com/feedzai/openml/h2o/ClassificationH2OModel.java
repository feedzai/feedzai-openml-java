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

import com.feedzai.openml.util.data.encoding.EncodingHelper;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.model.ClassificationMLModel;
import com.feedzai.openml.model.MachineLearningModel;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import hex.ModelCategory;
import hex.genmodel.GenModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.exception.PredictException;
import hex.genmodel.easy.prediction.AbstractPrediction;
import hex.genmodel.easy.prediction.BinomialModelPrediction;
import hex.genmodel.easy.prediction.MultinomialModelPrediction;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.SortedSet;

/**
 * Classification object used to represent a {@link MachineLearningModel model} generated in H2O.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public class ClassificationH2OModel implements ClassificationMLModel {

    /**
     * Logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(ClassificationH2OModel.class);

    /**
     * A wrapper for the model generated in H2O.
     */
    private final EasyPredictModelWrapper modelWrapper;

    /**
     * The path from where the model was initially loaded.
     */
    private final Path modelPath;

    /**
     * The {@link DatasetSchema} the model uses.
     */
    private final DatasetSchema schema;

    /**
     * A {@link Closeable} that needs to be closed upon {@link #close()}.
     */
    private final Closeable closeable;

    /**
     * Lock applied to {@code this.modelWrapper#predict}. Note that this is a workaround for PULSEDEV_24380.
     */
    private final Object predictLock;

    /**
     * Constructor for a {@link ClassificationH2OModel}.
     * @param genModel       The imported model generated in H2O.
     * @param modelPath      The path from where the model was initially loaded.
     * @param schema         The {@link DatasetSchema} the model uses.
     * @param closeable      A {@link Closeable} that needs to be closed upon {@link #close()}.
     */
    ClassificationH2OModel(final GenModel genModel,
                           final Path modelPath,
                           final DatasetSchema schema,
                           final Closeable closeable) {
        this.predictLock = new Object();
        this.modelWrapper = new EasyPredictModelWrapper(new EasyPredictModelWrapper.Config()
                                                                .setModel(genModel)
                                                                .setConvertUnknownCategoricalLevelsToNa(true));
        this.modelPath = Preconditions.checkNotNull(modelPath, "path of the model cannot be null");
        this.schema = Preconditions.checkNotNull(schema, "dataset schema cannot be null");
        this.closeable = Preconditions.checkNotNull(closeable, "the closeable cannot be null");

    }

    @Override
    public double[] getClassDistribution(final Instance instance) {
        final AbstractPrediction abstractPrediction = predictInstance(instance);

        final double[] classDistribution;
        if (isMultiClassification()) {
            classDistribution = ((MultinomialModelPrediction) abstractPrediction).classProbabilities;
        } else {
            classDistribution = ((BinomialModelPrediction) abstractPrediction).classProbabilities;
        }
        return convertDistribution(classDistribution);
    }

    @Override
    public int classify(final Instance instance) {
        final AbstractPrediction abstractPrediction = predictInstance(instance);

        final int predictedClass;
        if (isMultiClassification()) {
            predictedClass = ((MultinomialModelPrediction) abstractPrediction).labelIndex;
        } else {
            predictedClass = ((BinomialModelPrediction) abstractPrediction).labelIndex;
        }
        return convertClassification(predictedClass);
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
        this.closeable.close();
    }

    /**
     * Predicts the class of an {@link Instance}.
     *
     * @param instance The {@link Instance} to be classified.
     * @return the results of the classification.
     */
    private AbstractPrediction predictInstance(final Instance instance) {
        final RowData convertedInstance = convertInstanceToRowData(instance);
        final AbstractPrediction predict;
        try {
            // The sincronization here is a safe approach, since we don't know exactly where the non thread safe code is, only that it is downstream.
            // Also, note that from here further we are interacting in H2O's code, thus leaving the scope of this project.
            synchronized (this.predictLock) {
                predict = this.modelWrapper.predict(convertedInstance, this.modelWrapper.getModelCategory());
            }
        } catch (final PredictException exception) {
            throw new RuntimeException(
                    String.format("The model failed to classify the event[%s]!", convertedInstance),
                    exception
            );
        }

        return predict;
    }

    /**
     * Gets a list with the target values defined in the {@link #schema data schema} used by the model.
     *
     * @return a list with the target values used by the model.
     */
    private SortedSet<String> getTargetValues() {
        final FieldSchema targetField = this.schema.getTargetFieldSchema();
        return ((CategoricalValueSchema) targetField.getValueSchema()).getNominalValues();
    }

    /**
     * Converts the data type of an event in order to be classified by the H2O model.
     * It converts from {@link Instance} to {@link RowData}.
     *
     * @param instance The {@link Instance} to be classified.
     * @return The {@link RowData} to be classified.
     */
    private RowData convertInstanceToRowData(final Instance instance) {
        final RowData rowData = new RowData();

        final List<FieldSchema> fieldSchemas = this.schema.getFieldSchemas();
        FieldSchema fieldSchema;
        for (int i = 0; i < fieldSchemas.size(); i++) {
            fieldSchema = fieldSchemas.get(i);

            // If the schema of the field is categorical then we need to get its real value (in this case the value
            // stored in the instance represents the index of the value).
            if (fieldSchema.getValueSchema() instanceof CategoricalValueSchema) {
                final String instanceValue = EncodingHelper.decodeDoubleToCategory(
                        instance.getValue(i),
                        (CategoricalValueSchema) fieldSchema.getValueSchema()
                );
                rowData.put(fieldSchema.getFieldName(), instanceValue);
            } else {
                rowData.put(fieldSchema.getFieldName(), instance.getValue(i));
            }
        }
        return rowData;
    }

    /**
     * Converts the distribution values for a prediction which are in accordance with the schema used internally by the
     * model to the schema defined to be used by the model.
     *
     * @param distributionValuesModel The distribution values in accordance with the schema used internally by the model.
     * @return The distribution values in accordance with the schema defined to be used by the model.
     */
    private double[] convertDistribution(final double[] distributionValuesModel) {
        final SortedSet<String> targetValues = getTargetValues();
        final double[] distributionRealSchema = new double[targetValues.size()];

        final String[] modelTargetValues = this.modelWrapper.m.getDomainValues(this.modelWrapper.m.getResponseIdx());

        for (int i = 0; i < modelTargetValues.length; i++) {
            final String targetFeatureValue = modelTargetValues[i];
            final int indexModelTargetValue = Iterables.indexOf(targetValues, targetFeatureValue::equals);

            if (indexModelTargetValue == -1) {
                final String errorMsg = String.format("Unexpected value found: %s. Feature domain: %s", targetFeatureValue, targetValues);
                logger.error(errorMsg);
                throw new RuntimeException(errorMsg);
            }

            distributionRealSchema[indexModelTargetValue] = distributionValuesModel[i];
        }
        return distributionRealSchema;
    }

    /**
     * Converts the classification value for a prediction which are in accordance with the schema used internally by the
     * model to the schema defined to be used by the model.
     *
     * @param classificationModelIndex Index of the classification value in accordance with the schema used internally
     *                                 by the model.
     * @return Index of the classification value in accordance with the schema defined to be used by the model.
     */
    private int convertClassification(final int classificationModelIndex) {
        final SortedSet<String> targetValues = getTargetValues();
        final String classificationModelValue = this.modelWrapper.getResponseDomainValues()[classificationModelIndex];
        return Iterables.indexOf(targetValues, classificationModelValue::equals);
    }

    /**
     * Identifies the model is multi classifier or not. A multi classifier model allows to predict the value of a
     * categorical field with more than two domain values.
     *
     * @return True if the model is multi classifier, false otherwise.
     */
    private boolean isMultiClassification() {
        return this.modelWrapper.getModelCategory() == ModelCategory.Multinomial;
    }
}
