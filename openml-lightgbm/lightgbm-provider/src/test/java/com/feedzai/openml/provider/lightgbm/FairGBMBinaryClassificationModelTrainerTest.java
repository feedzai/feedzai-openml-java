package com.feedzai.openml.provider.lightgbm;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.BeforeClass;
import org.junit.Test;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.provider.exception.ModelLoadingException;

import static org.assertj.core.api.Assertions.assertThat;

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
    private final static Map<String, String> MODEL_PARAMS = TestParameters.getDefaultFairGBMParameters();

    /**
     * Maximum number of instances to train (to speed up tests).
     */
    private final static int MAX_NUMBER_OF_INSTANCES_TO_TRAIN = (int) 5e3;

    /**
     * Maximum number of instances to score (to speed up tests).
     */
    private final static int MAX_NUMBER_OF_INSTANCES_TO_SCORE = 300;

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
        MODEL_PARAMS.replace("constraint_group_column", "name:sensitive_group");

        // Override number of iterations in fit tests for faster tests:
//        MODEL_PARAMS.replace(NUM_ITERATIONS_PARAMETER_NAME, NUM_ITERATIONS_FOR_FAST_TESTS);
    }

//    @Test
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
