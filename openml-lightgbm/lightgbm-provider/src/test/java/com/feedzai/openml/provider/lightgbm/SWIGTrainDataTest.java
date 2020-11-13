package com.feedzai.openml.provider.lightgbm;

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class SWIGTrainDataTest {

    /**
     * SWIGTrainData instance common to all tests.
     * Fresh instance initialized before each test.
     */
    private SWIGTrainData swigTrainData;

    /**
     * Number of features for model.
     */
    private long numFeatures = 5;

    /**
     * Number of instances to store per chunk
     */
    private long numInstancesPerChunk = 3;

    /**
     * Set up the LightGBM libs.
     */
    @BeforeClass
    public static void setupFixture() {
        LightGBMUtils.loadLibs(); // Initialize LightGBM libs.
    }

    /**
     * Set up a fresh SWIGTrainResources test instance for every test.
     */
    @Before
    public void setupTest() {
        swigTrainData = new SWIGTrainData((int)numFeatures, numInstancesPerChunk);
    }

    /**
     * Test SWIGTrainResources() - some public members should be initialized.
     */
    @Test
    public void constructorInitializesPublicMembers() {
        assertThat(swigTrainData.swigOutDatasetHandlePtr).as("swigOutDatasetHandlePtr").isNotNull();

        assertThat(swigTrainData.swigLabelsChunkedArray.get_chunks_count())
                .as("swigLabelsChunkedArray.get_chunks_count()")
                .isOne();
        assertThat(swigTrainData.swigLabelsChunkedArray.get_added_count())
                .as("swigLabelsChunkedArray.get_added_count()")
                .isZero();

        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunks_count())
                .as("swigFeaturesChunkedArray.get_chunks_count()")
                .isOne();
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_added_count())
                .as("swigFeaturesChunkedArray.get_added_count()")
                .isZero();

        /* Cannot assert this as it requires external initialization:
         assertThat(swigTrainResources.swigDatasetHandle).as("swigDatasetHandle").isNotNull();
        */
    }

    /**
     * Test close() - All public fields should be freed and set to null.
     */
    @Test
    public void closeResetsAllPublicMembers() {

        swigTrainData.swigLabelsChunkedArray.add(1);
        swigTrainData.swigFeaturesChunkedArray.add(1);
        swigTrainData.releaseResources();

        assertThat(swigTrainData.swigOutDatasetHandlePtr).as("swigOutDatasetHandlePtr").isNull();
        assertThat(swigTrainData.swigTrainLabelDataArray).as("swigTrainLabelDataArray").isNull();
        assertThat(swigTrainData.swigDatasetHandle).as("swigDatasetHandle").isNull();

        assertThat(swigTrainData.swigLabelsChunkedArray.get_chunks_count())
                .as("swigLabelsChunkedArray.get_chunks_count()")
                .isZero();
        assertThat(swigTrainData.swigLabelsChunkedArray.get_added_count())
                .as("swigLabelsChunkedArray.get_added_count()")
                .isZero();

        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunks_count())
                .as("swigFeaturesChunkedArray.get_chunks_count()")
                .isZero();
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_added_count())
                .as("swigFeaturesChunkedArray.get_added_count()")
                .isZero();
    }

}
