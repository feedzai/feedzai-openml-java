package com.feedzai.openml.provider.lightgbm;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.provider.exception.ModelLoadingException;

import static com.feedzai.openml.provider.lightgbm.FairGBMDescriptorUtil.CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME;
import static com.feedzai.openml.provider.lightgbm.LightGBMBinaryClassificationModelTrainerTest.average;
import static com.feedzai.openml.provider.lightgbm.LightGBMBinaryClassificationModelTrainerTest.ensureFeatureContributions;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Tests for using the LightGBMBinaryClassificationModelTrainer class with FairGBM.
 *
 * @author Andre Cruz (andre.cruz@feedzai.com)
 * @since 1.2.0 // TODO: check which is the current version
 */
public class FairGBMBinaryClassificationModelTrainerTest {

    /**
     * Parameters for model train.
     */
    private static final Map<String, String> MODEL_PARAMS = TestParameters.getDefaultFairGBMParameters();

    /**
     * Maximum number of instances to train (to speed up tests).
     */
    private static final int MAX_NUMBER_OF_INSTANCES_TO_TRAIN = (int) 5e3;

    /**
     * Maximum number of instances to score (to speed up tests).
     */
    private static final int MAX_NUMBER_OF_INSTANCES_TO_SCORE = 300;

    /**
     * Dataset resource name to use for both fit and validation stages during tests.
     */
    static final String DATASET_RESOURCE_NAME = "test_data/in_train_val.csv";

    /**
     * The number of iterations used to test model fit.
     * The smaller, the faster the tests go, but make sure you're testing anything still.
     */
    private static final String NUM_ITERATIONS_FOR_FAST_TESTS = "2";

    /**
     * For unit tests, as train data is smaller, using the default chunk sizes to train
     * would mean that performing fits with multiple chunks would not be tested.
     * Hence, all model score tests are done with smaller chunk sizes to ensure
     * fitting with chunked data works.
     */
    public static final int SMALL_TRAIN_DATA_CHUNK_INSTANCES_SIZE = 221;

    /**
     * Load the LightGBM utils or nothing will work.
     * Also changes parameters.
     */
    @BeforeClass
    public static void setupFixture() {
        LightGBMUtils.loadLibs();

        // The constraint group column must be specified for the dataset used in tests
        MODEL_PARAMS.replace(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME, "name:sensitive_group");

        // Override the multiplier_learning_rate
        MODEL_PARAMS.replace("multiplier_learning_rate", "1000");

        // Override number of iterations in fit tests for faster tests:
//        MODEL_PARAMS.replace(NUM_ITERATIONS_PARAMETER_NAME, NUM_ITERATIONS_FOR_FAST_TESTS);
    }

    /**
     * Asserts that a model trained with numericals+categoricals and evaluated on the same datasource
     * has in average higher scores for the positive class (1) than for the negative one (0).
     *
     * @throws URISyntaxException    In case of error retrieving the data resource path.
     * @throws IOException           In case of error reading data.
     * @throws ModelLoadingException In case of error training the model.
     */
    @Test
    public void fitWithNumericalsAndCategoricals() throws URISyntaxException, IOException, ModelLoadingException {

        final ArrayList<List<Double>> scoresPerClass = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE,
                SMALL_TRAIN_DATA_CHUNK_INSTANCES_SIZE
        );

        assertThat(average(scoresPerClass.get(0))).as("score average per class")
                                                  .isLessThan(average(scoresPerClass.get(1)));
    }

    /**
     * Assert that in general, a model trained+scored on schemas where the position of
     * the label changes results in exactly the same scores.
     * <p>
     * This tests for regressions on the copying data code during train that at the start
     * of development resulted in broken scores (mostly constant) that were very hard to diagnose.
     *
     * @throws URISyntaxException    In case of error retrieving the data resource path.
     * @throws IOException           In case of error reading data.
     * @throws ModelLoadingException In case of error training the model.
     */
    @Test
    public void fitCategoricalsWithLabelInStartMiddleOrEndHasSameResults()
            throws URISyntaxException, IOException, ModelLoadingException {

        final ArrayList<List<Double>> scoresPerClassForLabelAtStart = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE,
                SMALL_TRAIN_DATA_CHUNK_INSTANCES_SIZE
        );

        final ArrayList<List<Double>> scoresPerClassForLabelInMiddle = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_IN_MIDDLE,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE,
                SMALL_TRAIN_DATA_CHUNK_INSTANCES_SIZE
        );

        final ArrayList<List<Double>> scoresPerClassForLabelAtEnd = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE,
                SMALL_TRAIN_DATA_CHUNK_INSTANCES_SIZE
        );

        assertThat(scoresPerClassForLabelAtStart).as("scores")
                                                 .isEqualTo(scoresPerClassForLabelInMiddle)
                                                 .isEqualTo(scoresPerClassForLabelAtEnd);
    }

    @Test
    public void fitResultsAreIndependentOfTrainChunkSizes()
            throws URISyntaxException, IOException, ModelLoadingException {

        final ArrayList<List<Double>> scoresWithSmallChunks = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE,
                15
        );

        final ArrayList<List<Double>> scoresWithChunks = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE,
                250
        );

        final ArrayList<List<Double>> scoresWithSingleChunk = fitModelAndGetFirstScoresPerClass(
                DATASET_RESOURCE_NAME,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                MAX_NUMBER_OF_INSTANCES_TO_TRAIN,
                MAX_NUMBER_OF_INSTANCES_TO_SCORE,
                10000000 // Try to have a single chunk (10M instances / chunk)
        );

        assertThat(scoresWithSmallChunks).as("scores")
                                         .isEqualTo(scoresWithChunks)
                                         .isEqualTo(scoresWithSingleChunk);
    }

    /**
     * Assert that there's an error when training with no instances.
     */
    @Test
    public void fitWithNoInstances() {

        final List<Instance> noInstances = new ArrayList<>();
        final Dataset emptyDataset = new MockDataset(TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START, noInstances);

        assertThatThrownBy(() ->
                new LightGBMModelCreator().fit(
                        emptyDataset,
                    new Random(),
                    MODEL_PARAMS
                )
        )
        .isInstanceOf(RuntimeException.class);
    }

    /**
     * Test Feature Contributions with target at end.
     *
     * @throws URISyntaxException For errors when loading the dataset resource.
     * @throws IOException        For errors when reading the dataset.
     * @since 1.3.0
     */
    @Test
    public void testFeatureContributionsTargetEnd() throws URISyntaxException, IOException {
        final Dataset dataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath(DATASET_RESOURCE_NAME),
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END,
                10000
        );
        ensureFeatureContributions(dataset, MODEL_PARAMS);
    }

    /**
     * Test Feature Contributions with target at middle.
     *
     * @throws URISyntaxException For errors when loading the dataset resource.
     * @throws IOException        For errors when reading the dataset.
     * @since 1.3.0
     */
    @Test
    public void testFeatureContributionsTargetMiddle() throws URISyntaxException, IOException {
        final Dataset dataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath(DATASET_RESOURCE_NAME),
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_IN_MIDDLE,
                10000
        );
        ensureFeatureContributions(dataset, MODEL_PARAMS);
    }

    /**
     * Test Feature Contributions with target at beginning.
     *
     * @throws URISyntaxException For errors when loading the dataset resource.
     * @throws IOException        For errors when reading the dataset.
     * @since 1.3.0
     */
    @Test
    public void testFeatureContributionsTargetBeginning() throws URISyntaxException, IOException {
        final Dataset dataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath(DATASET_RESOURCE_NAME),
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START,
                10000
        );
        ensureFeatureContributions(dataset, MODEL_PARAMS);
    }

//    @Test
    public void testFairnessFairGBMvsLightGBM() throws URISyntaxException, IOException {
        // TODO
        //  1. Train a FairGBM and a LightGBM model on the same data;
        //  2. Compute evaluations with breakdowns per sensitive group;
        //  3. compute fairness for each model;
        //  4. Assert FairGBM achieves x% higher fairness than LightGBM;
    }


    static ArrayList<List<Double>> fitModelAndGetFirstScoresPerClass(
            final String datasetResourceName,
            final DatasetSchema schema,
            final int maxInstancesToTrain,
            final int maxInstancesToScore,
            final int chunkSizeInstances) throws URISyntaxException, IOException, ModelLoadingException {

        return LightGBMBinaryClassificationModelTrainerTest.fitModelAndGetFirstScoresPerClass(
                datasetResourceName,
                schema,
                maxInstancesToTrain,
                maxInstancesToScore,
                chunkSizeInstances,
                MODEL_PARAMS
        );
    }
}
