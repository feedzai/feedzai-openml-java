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
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.model.ClassificationMLModel;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import mockit.Mocked;
import org.assertj.core.api.Assertions;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.SortedSet;
import java.util.concurrent.ThreadLocalRandom;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Tests for loading models with {@link DataRobotModelProvider}.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public class DataRobotModelProviderLoadTest extends AbstractDataRobotModelProviderLoadTest {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(DataRobotModelProviderLoadTest.class);

    /**
     * Verifies that the {@link ClassificationMLModel#classify(Instance)} " returns the index of the greatest value in
     * the class probability distribution produced by the calling
     * {@link ClassificationMLModel#getClassDistribution(Instance)} on the model
     *
     * @see ClassificationMLModel
     */
    @Test
    public void canGetClassDistributionMaxValueIndex() throws Exception {

        final ClassificationBinaryDataRobotModel model = getFirstModel();

        final Instance instance = getDummyInstance();

        this.canGetClassDistributionMaxValueIndex(model, instance);

    }

    /**
     * Checks that is possible to use compatible target values with the values used to train the model.
     *
     * @throws ModelLoadingException If the target are incompatible.
     */
    @Test
    public void compatibleTargetValuesWithSchema() throws ModelLoadingException {
        final Set<String> nominalValues = getFirstModelTargetNominalValues();
        final String[] targetModelValues = nominalValues.toArray(new String[]{});
        final SortedSet<String> sortedNominalValues = getFirstMachineLearningModelLoader()
                .checkTargetModelValuesWithSchema(
                        createDatasetSchema(nominalValues),
                        targetModelValues
                );

        assertThat(sortedNominalValues)
                .as("the target values used to train the model")
                .containsExactlyInAnyOrder(targetModelValues);
    }

    /**
     * Tests that a schema without target variable cannot be used in {@link DataRobotModelCreator#checkTargetModelValuesWithSchema(DatasetSchema, String[])}.
     */
    @Test
    public final void testCompatibleTargetValuesWithSchemaWithoutTargetVariable() {
        final Set<String> nominalValues = getFirstModelTargetNominalValues();
        final String[] targetModelValues = nominalValues.toArray(new String[]{});
        final DataRobotModelCreator modelCreator = getFirstMachineLearningModelLoader();
        final DatasetSchema schema = createDatasetSchema(nominalValues);

        assertThatThrownBy(() -> modelCreator.checkTargetModelValuesWithSchema(new DatasetSchema(schema.getFieldSchemas()), targetModelValues))
                .as("Checking target values against a schema with no target variable should fail.")
                .isInstanceOf(ModelLoadingException.class);
    }

    /**
     * Checks that is not possible to use incompatible target values with the values used to train the model.
     *
     * @throws ModelLoadingException If the target are incompatible.
     */
    @Test(expected = ModelLoadingException.class)
    public void incompatibleTargetValuesWithSchema() throws ModelLoadingException {
        final Set<String> nominalValues = getFirstModelTargetNominalValues();
        final String[] targetModelValues = new String[]{"true", "false"};
        getFirstMachineLearningModelLoader().checkTargetModelValuesWithSchema(
                createDatasetSchema(nominalValues),
                targetModelValues
        );
    }

    /**
     * Checks that is possible to use binary target values on DataRobot models.
     */
    @Test
    public void targetIsBinaryTest() {
        final Optional<List<ParamValidationError>> validationErrors = Optional.of(getFirstModelTargetNominalValues())
                .map(this::createDatasetSchema)
                .flatMap(DatasetSchema::getTargetFieldSchema)
                .map(FieldSchema::getValueSchema)
                .map(getFirstMachineLearningModelLoader()::validateTargetIsBinary);

        assertThat(validationErrors.get())
                .as("list of errors")
                .isEmpty();
    }

    /**
     * Checks that is not possible to use non-binary target values on DataRobot models.
     */
    @Test
    public void targetIsNotBinaryTest() {
        final MockDataset mockDataset = new MockDataset(
                ImmutableSet.of("c1", "c2", "c3"),
                10,
                1,
                new Random(123)
        );

        final Optional<List<ParamValidationError>> validationErrors = mockDataset.getSchema().getTargetFieldSchema()
                .map(FieldSchema::getValueSchema)
                .map(getFirstMachineLearningModelLoader()::validateTargetIsBinary);

        assertThat(validationErrors.get())
                .as("list of errors")
                .hasSize(1);
    }

    /**
     * Tests that Data Robot provider does not support models whose schema has no target variable.
     */
    @Test
    public final void testLoadModelWithNoTargetVariableFails() throws ModelLoadingException {
        final String modelPath = this.getClass().getResource("/boolean_classifier").getPath();
        final DataRobotModelCreator loader = getMachineLearningModelLoader(getValidAlgorithm());
        final DatasetSchema baseSchema = new DatasetSchema(loader.loadSchema(Paths.get(modelPath)).getFieldSchemas());

        assertThatThrownBy(() -> loader.loadModel(Paths.get(modelPath), baseSchema))
                .as("Loading a model by using a schema with no target variable will fail, as this is not supported by datarobot")
                .isInstanceOf(ModelLoadingException.class);
    }

    /**
     * Checks that it is possible to use a DataRobot binary classification model with various data-sets that have
     * different representations for the boolean target field.
     *
     * @throws ModelLoadingException If the model cannot be loaded.
     *
     * @since 0.5.2
     */
    @Test
    public void booleanTargetFieldTest() throws ModelLoadingException {
        final String modelPath = this.getClass().getResource("/boolean_classifier").getPath();
        final DataRobotModelCreator loader = getMachineLearningModelLoader(getValidAlgorithm());
        final DatasetSchema baseSchema = loader.loadSchema(Paths.get(modelPath));

        ImmutableList.of(
                ImmutableSet.of("True", "False"),
                ImmutableSet.of("False", "True"),
                ImmutableSet.of("true", "false"),
                ImmutableSet.of("false", "true"),
                ImmutableSet.of("True", "false"),
                ImmutableSet.of("false", "True"),
                ImmutableSet.of("true", "False"),
                ImmutableSet.of("False", "true"),
                ImmutableSet.of("TrUe", "FAlsE"),
                ImmutableSet.of("FAlsE", "TrUe"),
                ImmutableSet.of("TRUE", "FALSE"),
                ImmutableSet.of("FALSE", "TRUE")
        ).forEach(possibleClasses ->
                assertThatCode(() -> testPossibleBinaryValues(modelPath, loader, baseSchema, possibleClasses))
                        .doesNotThrowAnyException()
        );

        assertThatThrownBy(() -> testPossibleBinaryValues(modelPath, loader, baseSchema, ImmutableSet.of("banana", "laranja")))
                .as("The error thrown by the load")
                .isInstanceOf(ModelLoadingException.class)
                .hasMessageContaining("Incompatible target values")
                .hasMessageContaining("model is binary");
    }

    /**
     * Tests the possible given target values in a classification model.
     *
     * @param modelPath  The path to the model.
     * @param loader     The loader for the model.
     * @param baseSchema The schema to modify for the given target values.
     * @param classes    The possible target classes.
     * @throws ModelLoadingException If any error occurs.
     * @since 0.5.2
     */
    private void testPossibleBinaryValues(final String modelPath,
                                          final DataRobotModelCreator loader,
                                          final DatasetSchema baseSchema,
                                          final Set<String> classes) throws ModelLoadingException {
        logger.debug("Testing DR model for possible binary target values {}", classes);
        final DatasetSchema schema = cloneSchemaWithTarget(baseSchema, classes);
        testScoreModel(loader.loadModel(Paths.get(modelPath), schema), schema);
    }

    /**
     * Tests the score of a model with the given schema.
     *
     * @param model The model to test.
     * @param schema The schema to use.
     *
     * @since 0.5.2
     */
    private void testScoreModel(final ClassificationMLModel model, final DatasetSchema schema) {
        final double[] scores = model.getClassDistribution(new MockInstance(schema, new Random(321)));
        assertThat(Arrays.stream(scores).sum())
                .as("sum of the class distribution")
                .isCloseTo(1.0, Assertions.within(0.01));
    }

    /**
     * Clones the given schema to have a new target variable.
     *
     * @param oldSchema     The old schema to clone.
     * @param newTargetVals The values of the new target variable.
     * @return The new schema.
     * @since 0.5.2
     */
    private DatasetSchema cloneSchemaWithTarget(final DatasetSchema oldSchema, final Set<String> newTargetVals) {
        final FieldSchema oldTargetVar = oldSchema.getTargetFieldSchema()
                .orElseThrow(() -> new IllegalArgumentException("This schema must contain a target field"));
        final FieldSchema newTargetVar =
                new FieldSchema(oldTargetVar.getFieldName(), 0, new CategoricalValueSchema(false, newTargetVals));

        return new DatasetSchema(
                0,
                ImmutableList.<FieldSchema>builder()
                        .add(newTargetVar)
                        .addAll(oldSchema.getFieldSchemas().subList(1, oldSchema.getFieldSchemas().size()))
                        .build()
        );
    }

    /**
     * Checks that is possible to use Jar files to import DataRobot models.
     */
    @Test
    public void validModelFileFormatTest() {
        final Path modelPath = Paths.get(getClass().getResource("/" + getValidModelDirName()).getPath());
        final List<ParamValidationError> validationErrors = getFirstMachineLearningModelLoader()
                .validateModelFileFormat(modelPath);

        assertThat(validationErrors)
                .as("list of errors")
                .isEmpty();
    }

    /**
     * Checks that is not possible to use invalid file formats to import DataRobot models.
     */
    @Test
    public void invalidModelFileFormatTest() throws IOException {
        final File tempFile = File.createTempFile("temp-file-name", ".tmp");
        tempFile.deleteOnExit();

        final List<ParamValidationError> validationErrors = getFirstMachineLearningModelLoader()
                .validateModelFileFormat(tempFile.toPath());

        assertThat(validationErrors)
                .as("list of errors")
                .hasSize(1);
    }

    /**
     * Checks that it is possible to get debug information for the user in the error yielded when there is a schema
     * mis-match in the load of a model.
     *
     * @since 0.5.1
     */
    @Test
    public void schemaMismatchDebugInfoTest() {
        final String modelPath = this.getClass().getResource("/" + SECOND_MODEL_FILE).getPath();

        final DatasetSchema datasetSchema = new DatasetSchema(0, ImmutableList.of(
                new FieldSchema("field0", 0, new CategoricalValueSchema(false, ImmutableSet.of("0", "1"))),
                new FieldSchema("field1", 1, new NumericValueSchema(false)))
        );

        assertThatThrownBy(() -> this.getMachineLearningModelLoader(getValidAlgorithm()).loadModel(Paths.get(modelPath), datasetSchema))
                .isInstanceOf(ModelLoadingException.class)
                .hasMessageContaining("Wrong number of fields")
                .hasMessageContaining("expected 33")
                .hasMessageContaining("2 fields only");
    }

    /**
     * Tests that the failure to score generates an error.
     *
     * @throws ModelLoadingException If the model cannot be loaded.
     *
     * @since 0.5.8
     */
    @Test
    public void failedScoringTest() throws ModelLoadingException {
        final ClassificationBinaryDataRobotModel model = getFirstModel();

        // This causes the scoring to fail but also the logging to fail to decode the instance.
        final double[] mockedValues = model.getSchema().getFieldSchemas().stream()
                .map(field -> ThreadLocalRandom.current().nextDouble())
                .mapToDouble(val -> val)
                .toArray();

        assertThatThrownBy(() -> model.getClassDistribution(new MockInstance(mockedValues)))
                .as("The expected error during scoring")
                .hasMessageContaining("failed to classify")
                .hasMessageContaining("wrong number of features or wrong types");

        // Now we do the same but with an instance that is decodable for logging of the failure.
        final double mockedValue = 308120381203801238.0;
        final MockInstance correctSchemaInstance = new MockInstance(model.getSchema(), ThreadLocalRandom.current()) {
            @Override
            public double getValue(final int index) {
                if (index == 24) {
                    return mockedValue;  // Not a valid categorical for this field.
                }
                return super.getValue(index);
            }
        };

        assertThatThrownBy(() -> model.getClassDistribution(correctSchemaInstance))
                .as("The expected error during scoring")
                .hasMessageContaining("failed to classify")
                .hasMessageContaining(String.valueOf(mockedValue));
    }

    /**
     * Tests that fetching the labels from a model that does not contain them yields a default value.
     *
     * @param mockedPredictor The mocked predictor.
     *
     * @since 0.5.8
     */
    @Test
    public void readUnexistingTargetLabelsTest(final @Mocked Predictor mockedPredictor) {
        assertThat(new DataRobotModelCreator().getTargetModelValues(mockedPredictor))
                .as("The target labels")
                .containsExactlyInAnyOrder("0", "1");
    }
}
