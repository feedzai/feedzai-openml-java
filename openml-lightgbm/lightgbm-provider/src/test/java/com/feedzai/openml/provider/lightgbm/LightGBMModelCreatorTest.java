/*
 * Copyright 2020 Feedzai
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

package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.google.common.collect.ImmutableList;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.feedzai.openml.provider.lightgbm.LightGBMModelCreator.ERROR_MSG_NON_BINARY_TARGET;
import static com.feedzai.openml.provider.lightgbm.LightGBMModelCreator.ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE;
import static com.feedzai.openml.provider.lightgbm.LightGBMModelCreator.ERROR_MSG_RANDOM_FOREST_REQUIRES_BAGGING;
import static com.feedzai.openml.provider.lightgbm.LightGBMModelCreator.ERROR_MSG_SCHEMA_HAS_STRING_FIELDS;
import static com.feedzai.openml.provider.lightgbm.LightGBMModelCreator.ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELD_NAMES;
import static com.feedzai.openml.provider.lightgbm.LightGBMModelCreator.ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELDS_SIZE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;


/**
 * Tests for the LightGBMModelLoader.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
public class LightGBMModelCreatorTest {

    /**
     * The instance under test.
     */
    static private LightGBMModelCreator modelLoader;

    /**
     * Empty temporary dir path.
     */
    static private Path tmpEmptyDir;

    /**
     * No parameters to pass for now.
     */
    private final Map<String, String> params = new HashMap<>();

    /**
     * Prepares the tests.
     * @throws IOException in case creating the empty directory fails.
     */
    @BeforeClass
    static public void fixtureSetup() throws IOException {

        modelLoader = new LightGBMModelCreator();
        tmpEmptyDir = Files.createTempDirectory("tmp_empty_lgbm_tests_dir_");
    }

    /**
     * Tests teardown procedure.
     * @throws IOException in case the temporary directory cannot be deleted.
     */
    @AfterClass
    static public void fixtureTearDown() throws IOException {
        Files.delete(tmpEmptyDir);
    }

    /**
     * Ensure we can't load a model that's not a LightGBM model.
     */
    @Test
    public void loadModelFailsOnNonModelResourceLoadTest() {

        assertThatExceptionOfType(ModelLoadingException.class).isThrownBy(() ->
            modelLoader.loadModel(
                    TestResources.getScoredInstancesPath(), // Shouldn't be able to load a non-LightGBM file as a model.
                    TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END
            ));

    }

    /**
     * Assert that loading a valid model in the proper path with the correct arguments returns no errors.
     *
     * @throws URISyntaxException In case resource files couldn't be retrieved.
     */
    @Test
    public void validateForLoadOnValidModelHasNoErrorsTest() throws URISyntaxException {

        final List<ParamValidationError> errors = modelLoader.validateForLoad(
                TestResources.getModelFilePath(),
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                this.params
        );

        assertZeroErrors(errors);
    }

    /**
     * Assert that loading a model from a model folder with the proper
     * filenames and correct arguments returns no errors.
     * <p>
     * The folder must contain a file which is the model in a specific location.
     *
     * @throws URISyntaxException In case resource files couldn't be retrieved.
     * @see LightGBMModelCreator#MODEL_BINARY_RESOURCE_FILE_NAME
     */
    @Test
    public void validateForLoadOnValidModelFolderHasNoErrorsTest() throws URISyntaxException {

        final List<ParamValidationError> errors = modelLoader.validateForLoad(
                TestResources.getModelFolderPath(),
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                params
        );

        assertZeroErrors(errors);
    }

    /**
     * Asserts that an error is returned if the imported model has more than 3 classes.
     * @throws URISyntaxException For errors extracting the model resource path.
     */
    @Test
    public void validateForLoadWithTargetWithMoreThan2ClassesGivesErrorTest() throws URISyntaxException {

        final List<ParamValidationError> errors = modelLoader.validateForLoad(
                TestResources.getModelFolderPath(),
                TestSchemas.SCHEMA_WITH_NON_BINARY_CLASSIFICATION_TARGET,
                params
        );

        assertSingleErrorMessage(errors, ERROR_MSG_NON_BINARY_TARGET);
    }

    /**
     * Assert that validateForLoad returns an error if string fields are used.
     *
     * @throws URISyntaxException In case there's an error retrieving the model resource.
     */
    @Test
    public void validateForLoadWithModelTrainedWithStringsGivesErrorTest() throws URISyntaxException {

        final DatasetSchema schemaWithStringField = new DatasetSchema(
                0,
                ImmutableList.of(
                        SchemaUtils.getFieldCopyWithIndex(TestSchemas.FRAUD_LABEL_INDEXED_FIELD, 0),
                        new FieldSchema("stringField", 1, new StringValueSchema(false))
                )
        );

        final List<ParamValidationError> errors = modelLoader.validateForLoad(
                TestResources.getModelFolderPath(),
                schemaWithStringField,
                this.params
        );

        assertSingleErrorMessage(errors, ERROR_MSG_SCHEMA_HAS_STRING_FIELDS);
    }

    /**
     * Assert that one can't train with a regression target.
     * @throws URISyntaxException in case retrieving the model resource fails.
     */
    @Test
    public void validateForLoadRegressionTargetReturnsErrorsTest() throws URISyntaxException {

        final List<ParamValidationError> errors = modelLoader.validateForLoad(
                TestResources.getModelFolderPath(),
                TestSchemas.SCHEMA_WITH_REGRESSION_TARGET,
                TestParameters.getDefaultLightGBMParameters()
        );

        assertThat(errors.size()).as("error count").isPositive();
    }

    /**
     * Assert that loading a model from a model, where the model file inside doesn't have the name
     * LightGBMDescriptorUtil#MODEL_BINARY_RESOURCE_FILE_NAME returns an error.
     *
     * @throws URISyntaxException In case resource files couldn't be retrieved.
     * @throws IOException        In case copying the model to generate the bad model folder fails.
     * @see LightGBMModelCreator#MODEL_BINARY_RESOURCE_FILE_NAME
     */
    @Test
    public void validateForLoadOnFolderWithWrongModelFileNameHasErrorsTest() throws URISyntaxException, IOException {

        // Generate temporary folder with a model file inside that is not properly named.
        final Path tmpBadModelDir = Files.createTempDirectory("LightGBM_test_model_folder_");
        final String badModelFileName = "a_wrong_model_name_" + LightGBMModelCreator.MODEL_BINARY_RESOURCE_FILE_NAME;
        final Path tmpModelPath = Files.copy(
                TestResources.getModelFilePath(),
                tmpBadModelDir.resolve(badModelFileName)
        );

        try {
            // Assert that loading a folder with the wrong model filename inside gives an error.
            final List<ParamValidationError> errors = modelLoader.validateForLoad(
                    tmpBadModelDir,
                    TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                    params
            );

            assertSingleErrorMessageWithPrefix(errors, ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE);
        } finally {
            Files.delete(tmpModelPath);
            Files.delete(tmpBadModelDir);
        }
    }

    /**
     * Assert that loading a folder doesn't work.
     *
     * @throws URISyntaxException In case resource files couldn't be retrieved.
     */
    @Test
    public void validateForLoadNoModelInPathReturnsErrorTest() throws URISyntaxException {

        final Path anyFolderPath = TestResources.getResourcePath("");

        final List<ParamValidationError> errors = modelLoader.validateForLoad(
                anyFolderPath,
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                params
        );

        assertSingleErrorMessageWithPrefix(errors, ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE);
    }

    /**
     * Assert that loading a path where a model file doesn't exist returns an error.
     *
     * @throws URISyntaxException In case resource files couldn't be retrieved.
     */
    @Test
    public void validateForLoadNoModelFileInPathReturnsErrorTest() throws URISyntaxException {

        final Path invalidModelPath = TestResources.getResourcePath("").resolve("_no_file_named_like_this_");

        final List<ParamValidationError> errors = modelLoader.validateForLoad(
                invalidModelPath,
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                this.params
        );

        assertSingleErrorMessageWithPrefix(errors, ERROR_MSG_PREFIX_CANNOT_FIND_MODEL_FILE);
    }

    /**
     * Assert that loading a model with a wrong number
     * of fields throws a ModelLoadingException.
     */
    @Test
    public void loadModelWithWrongNumberOfFieldsThrowsTest() {

        assertThatExceptionOfType(ModelLoadingException.class).isThrownBy(() ->
            modelLoader.loadModel(
                    TestResources.getModelFilePath(),
                    TestSchemas.BAD_NUMERICALS_SCHEMA_WITH_MISSING_FIELDS
            )
        ).withMessage(ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELDS_SIZE);
    }

    /**
     * Assert that loading a model with a wrong field
     * names order throws a ModelLoadingException.
     *
     * @since 1.0.18
     */
    @Test
    public void loadModelWithWrongFieldNamesOrderThrowsTest() {

        assertThatExceptionOfType(ModelLoadingException.class).isThrownBy(() ->
                modelLoader.loadModel(
                        TestResources.getModelFilePath(),
                        TestSchemas.BAD_NUMERICALS_SCHEMA_WITH_WRONG_FEATURES_ORDER
                )
        ).withMessage(ERROR_MSG_SCHEMA_WITH_WRONG_PREDICTIVE_FIELD_NAMES);
    }

    /**
     * Assert that loading a model folder with the model file inside with the proper name doesn't throw.
     *
     * @see LightGBMModelCreator#MODEL_BINARY_RESOURCE_FILE_NAME
     */
    @Test
    public void loadModelFolderDoesNotThrowTest() {

        assertThatCode(() ->
            modelLoader.loadModel(
                    TestResources.getModelFolderPath(),
                    TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END
            )
        ).doesNotThrowAnyException();
    }

    /**
     * Assert that loading a model by giving the model file path works.
     */
    @Test
    public void loadModelFileDoesNotThrowTest() {

        assertThatCode(() ->
            modelLoader.loadModel(
                    TestResources.getModelFilePath(),
                    TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END
            )
        ).doesNotThrowAnyException();
    }

    /**
     * Assert that validate for fir doesn't return errors if inputs are Ok.
     */
    @Test
    public void validateForFitOkInputsReturnsNoErrorsTest() {

        final List<ParamValidationError> errors = modelLoader.validateForFit(
                tmpEmptyDir,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END,
                TestParameters.getDefaultLightGBMParameters()
        );

        assertZeroErrors(errors);
    }

    /**
     * Assert that one can't train with a non-binary classification target.
     */
    @Test
    public void validateForFitNonBinaryClassificationTargetReturnsErrorTest() {

        final List<ParamValidationError> errors = modelLoader.validateForFit(
                tmpEmptyDir,
                TestSchemas.SCHEMA_WITH_NON_BINARY_CLASSIFICATION_TARGET,
                TestParameters.getDefaultLightGBMParameters()
        );

        assertSingleErrorMessage(errors, ERROR_MSG_NON_BINARY_TARGET);
    }

    /**
     * Assert that one can't train with a regression target.
     */
    @Test
    public void validateForFitRegressionTargetReturnsErrorsTest() {

        final List<ParamValidationError> errors = modelLoader.validateForFit(
                tmpEmptyDir,
                TestSchemas.SCHEMA_WITH_REGRESSION_TARGET,
                TestParameters.getDefaultLightGBMParameters()
        );

        assertThat(errors.size()).as("error count").isPositive();
    }

    /**
     * Test for warning against not using bagging when boosting type is Random Forest.
     */
    @Test
    public void validateForFitRandomForestBaggingTest() {

        final Map<String, String> rfParams = TestParameters.getDefaultLightGBMParameters();
        rfParams.put(LightGBMDescriptorUtil.BOOSTING_TYPE_PARAMETER_NAME, "rf");

        final List<ParamValidationError> errors = modelLoader.validateForFit(
                tmpEmptyDir,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END,
                rfParams
        );

        assertSingleErrorMessage(errors, ERROR_MSG_RANDOM_FOREST_REQUIRES_BAGGING);
    }

    /**
     * Assert that one can't train without a target.
     */
    @Test
    public void validateForFitNoTargetReturnsErrorsTest() {

        final List<ParamValidationError> errors = modelLoader.validateForFit(
                tmpEmptyDir,
                TestSchemas.SCHEMA_WITH_NO_TARGET,
                TestParameters.getDefaultLightGBMParameters()
        );

        assertThat(errors.size()).as("error count").isPositive();
    }

    /**
     * Asserts that there are no errors.
     *
     * @param errors list of errors
     */
    private static void assertZeroErrors(final List<ParamValidationError> errors) {

        assertThat(errors).as("errors").isEmpty();
    }

    /**
     * Asserts that there is a single error and has a particular message.
     *
     * @param errors       List of validation errors
     * @param errorMessage error message
     */
    private void assertSingleErrorMessage(final List<ParamValidationError> errors,
                                         final String errorMessage) {

        assertThat(errors.size()).as("errors count").isEqualTo(1);
        assertThat(errors.get(0).getMessage()).as("error message")
                .isEqualTo(errorMessage);
    }

    /**
     * Asserts that there is a single error and has a particular message prefix.
     *
     * @param errors             List of validation errors
     * @param errorMessagePrefix error message prefix
     */
    private void assertSingleErrorMessageWithPrefix(final List<ParamValidationError> errors,
                                                    final String errorMessagePrefix) {

        assertThat(errors.size()).as("errors count").isEqualTo(1);
        assertThat(errors.get(0).getMessage()).as("error message")
                .startsWith(errorMessagePrefix);
    }
}
