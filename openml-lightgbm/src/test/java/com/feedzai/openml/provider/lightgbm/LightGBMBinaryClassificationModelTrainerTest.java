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

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockDataset;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.NUM_ITERATIONS_PARAMETER_NAME;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Tests for the LightGBMBinaryClassificationModelTrainer class.
 * <p>
 * Tests LightGBM model fitting procedures.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 */
public class LightGBMBinaryClassificationModelTrainerTest {

    /**
     * Parameters for model train.
     */
    private final static Map<String, String> modelParams = TestParameters.getDefaultParameters();

    /**
     * Maximum number of instances to train (to speed up tests).
     */
    private final static int MAX_NUMBER_OF_INSTANCES_TO_TRAIN = (int) 5e3;

    /**
     * Maximum number of instances to score (to speed up tests).
     */
    private final static int MAX_NUMBER_OF_INSTANCES_TO_SCORE = 100;

    /**
     * Dataset resource name to use for both fit & validation stages during tests.
     */
    static final String DATASET_RESOURCE_NAME = "test_data/in_train_val.csv";

    /**
     * The number of iterations used to test model fit.
     * The smaller, the faster the tests go, but make sure you're testing anything still.
     */
    private static final String NUM_ITERATIONS_FOR_FAST_TESTS = "2";

    /**
     * Load the LightGBM utils or nothing will work.
     * Also changes parameters.
     */
    @BeforeClass
    public static void setupFixture() {
        LightGBMUtils.loadLibs();

        // Override number of iterations in fit tests for faster tests:
        modelParams.replace(NUM_ITERATIONS_PARAMETER_NAME, NUM_ITERATIONS_FOR_FAST_TESTS);
    }

    /**
     * Asserts that a model trained with numericals only and evaluated on the same datasource
     * has in average higher scores for the positive class (1) than for the negative one (0).
     *
     * @throws URISyntaxException In case of error retrieving the data resource path.
     * @throws IOException        In case of error reading data.
     */
    @Test
    public void fitWithNumericalsOnlyTest() throws URISyntaxException, IOException {

        final ArrayList<ArrayList<Double>> scoresPerClass = fitModelAndGetFirstScoresPerClass(
                TestResources.SCORED_INSTANCES_NAME,
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE
        );

        assertThat(average(scoresPerClass.get(0))).as("score average per class")
                .isLessThan(average(scoresPerClass.get(1)));
    }

    /**
     * Asserts that a model trained with numericals+categoricals and evaluated on the same datasource
     * has in average higher scores for the positive class (1) than for the negative one (0).
     *
     * @throws URISyntaxException In case of error retrieving the data resource path.
     * @throws IOException        In case of error reading data.
     */
    @Test
    public void fitWithNumericalsAndCategoricalsTest() throws URISyntaxException, IOException {

        final ArrayList<ArrayList<Double>> scoresPerClass = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE
        );

        assertThat(average(scoresPerClass.get(0))).as("score average per class")
                .isLessThan(average(scoresPerClass.get(1)));
    }

    /**
     * Assert that in general, a model trained+scored on schemas where the position of
     * the label changes results in exactly the same scores.
     *
     * This tests for regressions on the copying data code during train that at the start
     * of development resulted in broken scores (mostly constant) that were very hard to diagnose.
     *
     * @throws URISyntaxException In case of error retrieving the data resource path.
     * @throws IOException        In case of error reading data.
     */
    @Test
    public void fitCategoricalsWithLabelInStartMiddleOrEndHasSameResultsTest()
            throws URISyntaxException, IOException {

        final ArrayList<ArrayList<Double>> scoresPerClassForLabelAtStart = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE
        );

        final ArrayList<ArrayList<Double>> scoresPerClassForLabelInMiddle = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_IN_MIDDLE,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE
        );

        final ArrayList<ArrayList<Double>> scoresPerClassForLabelAtEnd = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE
        );

        assertThat(scoresPerClassForLabelAtStart).as("scores")
                .isEqualTo(scoresPerClassForLabelInMiddle)
                .isEqualTo(scoresPerClassForLabelAtEnd);
    }

    /**
     * Assert that there's an error when training with no instances.
     */
    @Test
    public void fitWithNoInstancesTest() {

        final List<Instance> noInstances = new ArrayList<>();
        final Dataset emptyDataset = new MockDataset(TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END, noInstances);

        assertThatThrownBy(() -> {
                    final LightGBMBinaryClassificationModel model = new LightGBMModelCreator().fit(
                            emptyDataset,
                            new Random(),
                            modelParams
                    );
                })
                .isInstanceOf(RuntimeException.class);
    }

    /**
     * Assert that with very little data, the model training occurs faster for
     * there are no more trees to train (in here, we have just one output tree).
     *
     * @throws URISyntaxException For errors when loading the dataset resource.
     * @throws IOException        For errors when reading the dataset.
     */
    @Test
    public void fitWithInsufficientInstancesTest() throws URISyntaxException, IOException {

        final Dataset tinyDataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath(DATASET_RESOURCE_NAME),
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END,
                10
        );

        final int numRequestedTrainIterations = 50;
        final Map<String, String> trainParams = new HashMap<>(modelParams);
        trainParams.replace(NUM_ITERATIONS_PARAMETER_NAME, String.valueOf(numRequestedTrainIterations));

        final LightGBMBinaryClassificationModel model = new LightGBMModelCreator().fit(
                tinyDataset,
                new Random(),
                trainParams
        );

        assertThat(model.getBoosterNumIterations()).as("number of booster iterations")
                // Ensure we requested more iterations than we got (train stopped earlier):
                .isLessThan(numRequestedTrainIterations)
                // Very little data available. 1 iteration is enough for a fit with no more residuals:
                .isEqualTo(1);
    }


    /**
     * With a dataset and schema choice, fit a model and evaluate it on part or the full data.
     *
     * @param datasetResourceName Dataset to use for train and validation.
     * @param schema              Schema to use for train and validation.
     * @param maxInstancesToTrain Maximum number of instances to train.
     * @param maxInstancesToScore Maximum number of instances to score.
     * @return array(arrayScoresClass0, arrayScoresClass1) Array of scores per class (binary case).
     * @throws URISyntaxException For errors when loading the dataset resource.
     * @throws IOException        For errors when reading the dataset.
     */
    private static ArrayList<ArrayList<Double>> fitModelAndGetFirstScoresPerClass(
            final String datasetResourceName,
            final DatasetSchema schema,
            final int maxInstancesToTrain,
            final int maxInstancesToScore) throws URISyntaxException, IOException {

        final Dataset dataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath(datasetResourceName),
                schema,
                maxInstancesToTrain
        );

        final LightGBMBinaryClassificationModel model = new LightGBMModelCreator().fit(
                dataset,
                new Random(),
                modelParams
        );

        return getClassScores(dataset, model, maxInstancesToScore);
    }

    /**
     * Computes an array of 2 arrays.
     * One to hold the scores of the 0-th class and another to hold the scores of the 1-st class.
     *
     * @param dataset      Input validation dataset
     * @param model        Input model
     * @param maxInstances Maximum number of instances to score (to reduce testing time).
     * @return Array with arrays of class scores.
     */
    private static ArrayList<ArrayList<Double>> getClassScores(final Dataset dataset,
                                                               final LightGBMBinaryClassificationModel model,
                                                               final int maxInstances) {

        final int targetIndex = dataset.getSchema().getTargetIndex().get(); // We need a label for the tests.

        final ArrayList<ArrayList<Double>> classScores = new ArrayList<>();
        classScores.add(new ArrayList<>());
        classScores.add(new ArrayList<>());

        final Iterator<Instance> iterator = dataset.getInstances();
        for (int numInstances = 0; iterator.hasNext() && numInstances < maxInstances; ++numInstances) {
            final Instance instance = iterator.next();
            final int classIndex = (int) instance.getValue(targetIndex);
            final double[] scoreDistribution = model.getClassDistribution(instance);

            classScores.get(classIndex).add(scoreDistribution[1]);
        }

        return classScores;
    }

    /**
     * Returns the average of an array.
     *
     * @param inputArray Input array from which to compute the average.
     * @return Average
     */
    private double average(final ArrayList<Double> inputArray) {

        double sum = 0.0;
        for (final Double x : inputArray) {
            sum += x;
        }
        return (sum / inputArray.size());
    }
}
