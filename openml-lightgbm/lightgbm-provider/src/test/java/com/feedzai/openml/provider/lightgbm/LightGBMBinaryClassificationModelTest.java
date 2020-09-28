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

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.assertTrue;

/**
 * Tests for the BinaryLightGBMModel class.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
 */
public class LightGBMBinaryClassificationModelTest {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(LightGBMBinaryClassificationModelTest.class);

    /**
     * LightGBM's model loader instance. Will be used to load more model instances during tests.
     */
    static final LightGBMModelCreator modelLoader = new LightGBMModelCreator();

    /**
     * Standard model (import schema matches the outside model train schema fields' order, including label position).
     */
    static LightGBMBinaryClassificationModel model;

    /**
     * Set up the standard model.
     *
     * @throws ModelLoadingException Error during model load.
     * @throws URISyntaxException    Couldn't extract file resource.
     */
    @BeforeClass
    static public void setupFixture() throws ModelLoadingException, URISyntaxException {
        model = modelLoader.loadModel(TestResources.getModelFilePath(), TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END);
    }

    /**
     * Test that getSchema returns the received test schema at the loadModel.
     */
    @Test
    public void getSchemaEqualsReceivedSchemaAtLoadModelTest() {
        assertThat(model.getSchema()).as("schema").isEqualTo(TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END);
    }

    /**
     * Ensure the number of features in the model is correctly retrieved.
     */
    @Test
    public void getBoosterNumFeaturesTest() {
        assertThat(model.getBoosterNumFeatures()).as("number of booster features")
                .isEqualTo(TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END.getPredictiveFields().size());
    }

    /**
     * Ensure scores match to reference python scores.
     * The target was the last field during training.
     *
     * @throws IOException        In case the CSVParser creation fails.
     * @throws URISyntaxException In case retrieving file resources fails.
     */
    @Test
    public void getClassDistributionScoresWithTargetAtEndTest() throws IOException, URISyntaxException {

        final CSVParser csvParser = TestResources.getScoredInstancesCSVParser();
        final DatasetSchema schema = TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END;

        for (final CSVRecord record : csvParser) {
            assertTrue(modelScoreMatchesReferenceScore(this.model, schema, record));
        }
    }

    /**
     * Ensure scores match to reference python scores,
     * when the imported model's target field was in the middle of the features.
     *
     * @throws IOException           In case the CSVParser creation fails.
     * @throws URISyntaxException    In case retrieving file resources fails.
     * @throws ModelLoadingException Error loading the model.
     */
    @Test
    public void getClassDistributionScoresWithTargedInMiddleTest()
            throws IOException, URISyntaxException, ModelLoadingException {

        final CSVParser csvParser = TestResources.getScoredInstancesCSVParser();
        final DatasetSchema schema = TestSchemas.NUMERICALS_SCHEMA_WITH_TARGET_IN_MIDDLE;

        final LightGBMBinaryClassificationModel model = modelLoader.loadModel(TestResources.getModelFilePath(), schema);

        for (final CSVRecord record : csvParser) {
            assertTrue(modelScoreMatchesReferenceScore(model, schema, record));
        }
    }

    /**
     * Ensure scores match to reference python scores,
     * when the imported model's target field was the first field.
     *
     * @throws IOException           In case the CSVParser creation fails.
     * @throws URISyntaxException    In case retrieving file resources fails.
     * @throws ModelLoadingException Error loading the model.
     */
    @Test
    public void getClassDistributionScoresWithTargedAtStartTest()
            throws IOException, URISyntaxException, ModelLoadingException {

        final CSVParser csvParser = TestResources.getScoredInstancesCSVParser();
        final DatasetSchema schema = TestSchemas.NUMERICALS_SCHEMA_WITH_TARGET_AT_START;

        final LightGBMBinaryClassificationModel model = modelLoader.loadModel(TestResources.getModelFilePath(), schema);

        for (final CSVRecord record : csvParser) {
            assertTrue(modelScoreMatchesReferenceScore(model, schema, record));
        }
    }

    /**
     * Assert that scoring with a schema that has no labels gives the correct scores too.
     *
     * @throws IOException           In case the CSVParser creation fails.
     * @throws URISyntaxException    In case retrieving file resources fails.
     * @throws ModelLoadingException Error loading the model.
     */
    @Test
    public void getClassDistributionScoresWithNoLabelTest()
            throws IOException, URISyntaxException, ModelLoadingException {

        final CSVParser csvParser = TestResources.getScoredInstancesCSVParser();
        final DatasetSchema schema = SchemaUtils.getSchemaWithoutLabel(TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END);

        final LightGBMBinaryClassificationModel model = modelLoader.loadModel(TestResources.getModelFilePath(), schema);

        for (final CSVRecord record : csvParser) {
            assertTrue(modelScoreMatchesReferenceScore(model, schema, record));
        }
    }

    /**
     * Assert that scores are !!WRONG!! if the model import features order is wrong.
     * It works as a null test to the other score tests, ensuring that we're testing correctly.
     *
     * @throws IOException           In case the CSVParser creation fails.
     * @throws URISyntaxException    In case retrieving file resources fails.
     * @throws ModelLoadingException Error loading the model.
     */
    @Test
    public void getClassDistributionScoresWithWrongFeaturesOrderTest()
            throws IOException, URISyntaxException, ModelLoadingException {

        final CSVParser csvParser = TestResources.getScoredInstancesCSVParser();
        final DatasetSchema schema = TestSchemas.BAD_NUMERICALS_SCHEMA_WITH_WRONG_FEATURES_ORDER;

        final LightGBMBinaryClassificationModel model = modelLoader.loadModel(
                TestResources.getModelFilePath(),
                TestSchemas.NUMERICALS_SCHEMA_WITH_TARGET_AT_START
        );

        int wrongScoresCounter = 0;
        for (final CSVRecord record : csvParser) {
            if (! modelScoreMatchesReferenceScore(model, schema, record))
                wrongScoresCounter += 1;
        }

        assertThat(wrongScoresCounter).as("wrong scores counter").isGreaterThan(0);
    }

    /**
     * Tests if the scoring of an instance matches the reference score.
     * The instance is extracted from the CSVRecord
     * (which also contains the reference score)
     * according to the schema.
     *
     * @param model  Model that will score
     * @param schema Schema to use (controls CSVRecord extraction)
     * @param record Shall contain at least the features and refererence score (at field "score").
     * @return true if scores match, false if they don't.
     */
    private boolean modelScoreMatchesReferenceScore(final LightGBMBinaryClassificationModel model,
                                                    final DatasetSchema schema,
                                                    final CSVRecord record) {

        final Instance instance = CSVUtils.createDoublesInstanceFromCSVRecord(schema, record);
        final double referenceScore = Double.parseDouble(record.get("score"));

        final double[] scoreDistribution = model.getClassDistribution(instance);

        // Due to discrepancies in scores we compare up to the last 2 digits (non-included):
        return compareDoubles(referenceScore, scoreDistribution[1], 2);
    }

    /**
     * Compares two doubles, being equal if all but the last excludedDigits of the least precise double match.
     * @param a First double.
     * @param b Second double.
     * @param excludedDigits Exclude comparison of this many digits from the least precise number and compare.
     * @return true if and only if a == b (at the chosen precision level)
     */
    private boolean compareDoubles(final double a, final double b, final int excludedDigits) {

        final BigDecimal decimal_a = BigDecimal.valueOf(a);
        final BigDecimal decimal_b = BigDecimal.valueOf(b);

        final int smallerScale = Math.min(decimal_a.scale(), decimal_b.scale()) - excludedDigits;

        final RoundingMode roundingMode = RoundingMode.UP;

        return 0 == decimal_a.setScale(smallerScale, roundingMode).compareTo(
                decimal_b.setScale(smallerScale, roundingMode)
        );
    }

    /**
     * Compares model files in the standard_code_models/ and new_code_models/.
     * If there are any differences in the models within such folders they are reported
     * in a colored report (requires python3 with termcolor installed).
     *
     * @param referenceModelsFolder Folder with reference models
     * @param newModelsFolder       Folder with new models to compare to reference
     * @return true if there are no differences and all the dependencies are installed, false otherwise.
     * @throws IOException          In case the process errors.
     * @throws InterruptedException In case awaiting for the process to finish fails.
     * @since 1.0.19
     */
    private boolean compareModelFilesAndDoPrettyReport(
            final String referenceModelsFolder,
            final String newModelsFolder) throws IOException, InterruptedException {

        // Check for differences and report in detailed manner!
        final Process p = new ProcessBuilder(
                "python",
                "diff_models.py",
                referenceModelsFolder,
                newModelsFolder
        ).start();

        // Fetch process output
        final BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        final StringBuilder builder = new StringBuilder();
        String line = null;
        while ((line = reader.readLine()) != null) {
            builder.append(line);
            builder.append(System.getProperty("line.separator"));
        }
        final String processOutput = builder.toString();

        p.waitFor();
        logger.error(processOutput);

        return (p.exitValue() == 0);
    }

    /**
     * Asserts that the two files have the same content.
     *
     * @param name      name of the file to compare for the assert message.
     * @param filepath1 path to the first file
     * @param filepath2 path to the second file
     * @throws IOException Raised in case of failure reading the files
     * @since 1.0.19
     */
    private void assertEqualFileContents(final String name,
                                         final Path filepath1,
                                         final Path filepath2) throws IOException {

        final File file1 = new File(filepath1.toString());
        final File file2 = new File(filepath2.toString());

        assertThat(FileUtils.contentEquals(file1, file2))
                .as(String.format("%s file comparison", name))
                .isTrue();
    }

    /**
     * This functional test ensures that LightGBM can read a model file and output one exactly alike the one read in.
     * This is to ensure the new code to rewrite the model read/write layers is completely functional.
     * The two reference models were generated initially with LightGBM's v3.0.0 code.
     * The two generated ones will use the current code in the current locale. There should be no mismatches.
     *
     * @throws URISyntaxException    For invalid resource paths
     * @throws ModelLoadingException Errors when loading the model resources
     * @throws IOException           IO Errors opening/writing
     * @throws InterruptedException  Thrown if the model report fails to await for the process
     * @since 1.0.19
     */
    @Test
    public void testRewriteModel() throws URISyntaxException, ModelLoadingException, IOException, InterruptedException {

        final String referenceModelsFolder = "standard_code_models";
        final String testOutputModelsFolder = "new_code_models";
        final String model1Name = "4f.txt";
        final String model2Name = "42f.txt";

        // Create the output directory if it doesn't exist:
        new File(testOutputModelsFolder).mkdir();

        // Rewrite model 4f.txt:
        LightGBMSWIG swig = new LightGBMSWIG(
                TestResources.getModelFilePath().toString(),
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                "");
        swig.saveModelToDisk(Paths.get(testOutputModelsFolder, model1Name));

        // Rewrite model 42f.txt:
        swig = new LightGBMSWIG(
                TestResources.getResourcePath("lightgbm_model_42_numericals.txt").toString(),
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                "");
        swig.saveModelToDisk(Paths.get(testOutputModelsFolder, model2Name));

        // Do a detailed report (if Python3+termcolor is available):
        compareModelFilesAndDoPrettyReport(
                referenceModelsFolder,
                testOutputModelsFolder
        );

        // Compare the rewritten models:
        assertEqualFileContents(
                model1Name,
                Paths.get(referenceModelsFolder, model1Name),
                Paths.get(testOutputModelsFolder, model1Name));

        assertEqualFileContents(
                model1Name,
                Paths.get(referenceModelsFolder, model2Name),
                Paths.get(testOutputModelsFolder, model2Name));
    }
}
