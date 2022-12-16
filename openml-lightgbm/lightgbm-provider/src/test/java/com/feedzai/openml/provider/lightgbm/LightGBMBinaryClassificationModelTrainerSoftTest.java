package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.provider.exception.ModelLoadingException;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.feedzai.openml.provider.lightgbm.LightGBMBinaryClassificationModelTrainerTest.average;
import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.NUM_ITERATIONS_PARAMETER_NAME;
import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.SOFT_LABEL_PARAMETER_NAME;
import static com.feedzai.openml.provider.lightgbm.resources.schemas.SoftSchemas.SOFT_SCHEMA;
import static java.nio.file.Files.createTempDirectory;
import static org.assertj.core.api.Assertions.assertThat;

public class LightGBMBinaryClassificationModelTrainerSoftTest {

    /**
     * Parameters for model train.
     */
    private static final Map<String, String> MODEL_PARAMS = TestParameters.getDefaultLightGBMParameters();

    /**
     * Maximum number of instances to train (to speed up tests).
     */
    private static final int MAX_NUMBER_OF_INSTANCES_TO_TRAIN = 5000;

    /**
     * Maximum number of instances to score (to speed up tests).
     */
    private static final int MAX_NUMBER_OF_INSTANCES_TO_SCORE = 300;

    /**
     * Dataset resource name to use for both fit and validation stages during tests.
     */
    static final String DATASET_RESOURCE_NAME = "test_data/soft.csv";

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

        // Override number of iterations in fit tests for faster tests:
        MODEL_PARAMS.replace(NUM_ITERATIONS_PARAMETER_NAME, NUM_ITERATIONS_FOR_FAST_TESTS);
    }

    @Test
    public void fitWithSoftLabels() throws ModelLoadingException, URISyntaxException, IOException {

        final Map<String, String> trainParams = modelParamsWith(MODEL_PARAMS, SOFT_LABEL_PARAMETER_NAME, "soft");

        assertThat(new LightGBMModelCreator().validateForFit(createTempDirectory("lixo"), SOFT_SCHEMA, trainParams)).isEmpty();

        final ArrayList<List<Double>> scoresPerClass = LightGBMBinaryClassificationModelTrainerTest.fitModelAndGetFirstScoresPerClass(
                "test_data/soft.csv",
                SOFT_SCHEMA,
                9999,
                9999,
                100,
                trainParams
        );

        assertThat(average(scoresPerClass.get(1)) - average(scoresPerClass.get(0)))
                .isGreaterThan(0.1);
    }

    @Test
    public void fitWithSoftLabelsUninformativeHasNoDistinction() throws ModelLoadingException, URISyntaxException, IOException {

        final Map<String, String> trainParams = modelParamsWith(MODEL_PARAMS, SOFT_LABEL_PARAMETER_NAME, "soft_uninformative");

        assertThat(new LightGBMModelCreator().validateForFit(createTempDirectory("lixo"), SOFT_SCHEMA, trainParams)).isEmpty();

        final ArrayList<List<Double>> scoresPerClass = LightGBMBinaryClassificationModelTrainerTest.fitModelAndGetFirstScoresPerClass(
                "test_data/soft.csv",
                SOFT_SCHEMA,
                9999,
                9999,
                100,
                trainParams
        );

        assertThat(average(scoresPerClass.get(1)) - average(scoresPerClass.get(0)))
                .isLessThan(0.01);
    }

    public static Map<String, String> modelParamsWith(final Map<String, String> params, final String key, final String value) {
        final Map<String, String> trainParams = new HashMap<>();
        params
                .entrySet()
                .stream()
                .filter(entry -> !entry.getKey().equals(key))
                .forEach(entry -> trainParams.put(entry.getKey(), entry.getValue()));
        trainParams.put(key, value);
        return trainParams;
    }
}
