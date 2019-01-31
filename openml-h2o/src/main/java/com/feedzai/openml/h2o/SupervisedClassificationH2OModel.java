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

package com.feedzai.openml.h2o;

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import hex.ModelCategory;
import hex.genmodel.GenModel;
import hex.genmodel.easy.prediction.BinomialModelPrediction;
import hex.genmodel.easy.prediction.MultinomialModelPrediction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.SortedSet;

/**
 * A Classification Model representation for supervised algorithms.
 *
 * <p>
 *     Supervised Models require a {@link DatasetSchema schema} with target field.
 * </p>
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 */
public class SupervisedClassificationH2OModel extends AbstractClassificationH2OModel {

    /**
     * Logger for {@link SupervisedClassificationH2OModel}.
     */
    private static final Logger logger = LoggerFactory.getLogger(SupervisedClassificationH2OModel.class);

    /**
     * Constructor for a {@link AbstractClassificationH2OModel}.
     *
     * @param genModel  The imported model generated in H2O.
     * @param modelPath The path from where the model was initially loaded.
     * @param schema    The {@link DatasetSchema} the model uses.
     * @param closeable A {@link Closeable} that needs to be closed upon {@link #close()}.
     */
    SupervisedClassificationH2OModel(final GenModel genModel, final Path modelPath, final DatasetSchema schema, final Closeable closeable) {
        super(genModel, modelPath, schema, closeable);
        Preconditions.checkArgument(schema.getTargetFieldSchema().isPresent(), "Supervised models require a schema with target field.");
    }

    @Override
    public double[] getClassDistribution(final Instance instance) {
        final double[] classDistribution;
        if (isMultiClassification()) {
            classDistribution = this.<MultinomialModelPrediction>predictInstance(instance).classProbabilities;
        } else {
            classDistribution = this.<BinomialModelPrediction>predictInstance(instance).classProbabilities;
        }
        return convertDistribution(classDistribution);
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
                throw new IllegalStateException(errorMsg);
            }

            distributionRealSchema[indexModelTargetValue] = distributionValuesModel[i];
        }
        return distributionRealSchema;
    }

    /**
     * Gets a list with the target values defined in the {@link #schema data schema} used by the model.
     *
     * @return a list with the target values used by the model.
     */
    private SortedSet<String> getTargetValues() {
        return this.schema.getTargetFieldSchema()
                .map(FieldSchema::getValueSchema)
                .map(CategoricalValueSchema.class::cast)
                .map(CategoricalValueSchema::getNominalValues)
                .orElse(ImmutableSortedSet.of());
    }

    @Override
    public int classify(final Instance instance) {
        final int predictedClass;
        if (isMultiClassification()) {
            predictedClass = this.<MultinomialModelPrediction>predictInstance(instance).labelIndex;
        } else {
            predictedClass = this.<BinomialModelPrediction>predictInstance(instance).labelIndex;
        }
        return convertClassification(predictedClass);
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
     * Identifies if the model is multi classifier or not. A multi classifier model allows to predict the value of a
     * categorical field with more than two domain values.
     *
     * @return True if the model is multi classifier, false otherwise.
     */
    private boolean isMultiClassification() {
        return this.modelWrapper.getModelCategory() == ModelCategory.Multinomial;
    }
}
