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
    private long numInstancesPerChunk = 16;

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
     * Assert the features ChunkedArray has the proper size.
     */
    @Test
    public void featuresChunkedArraySize() {
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunk_size())
                .as("swigTrainData.swigFeaturesChunkedArray.get_chunk_size()")
                .isEqualTo(numFeatures * numInstancesPerChunk);
    }

    /**
     * The Features ChunkedArray size MUST be multiple of the instance size (numFeatures).
     * An instance cannot be broken across two partitions (chunks) when creating the
     * LightGBM Dataset.
     */
    @Test
    public void featuresChunkedArrayChunksHoldCompleteInstances() {
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunk_size() % numFeatures)
                .as("swigTrainData.swigFeaturesChunkedArray.get_chunk_size() % numFeatures")
                .isEqualTo(0);
    }

    /**
     * Ensure features' chunks can hold the requested number of features.
     */
    @Test
    public void featuresChunkedArrayHoldsRequestedInstances() {
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunk_size() / numFeatures)
                .as("swigTrainData.swigFeaturesChunkedArray.get_chunk_size() / numFeatures")
                .isEqualTo(numInstancesPerChunk);
    }

    /**
     * Not only features', but labels' ChunkedArray must hold chunks of the requested
     * `numInstancesPerChunk` size.
     */
    @Test
    public void labelChunkedArrayHoldsRequestedInstances() {
        assertThat(swigTrainData.swigLabelsChunkedArray.get_chunk_size())
                .as("swigTrainData.swigLabelsChunkedArray.get_chunk_size()")
                .isEqualTo(numInstancesPerChunk);
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
        assertThat(swigTrainData.swigLabelsChunkedArray.get_add_count())
                .as("swigLabelsChunkedArray.get_add_count()")
                .isZero();

        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunks_count())
                .as("swigFeaturesChunkedArray.get_chunks_count()")
                .isOne();
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_add_count())
                .as("swigFeaturesChunkedArray.get_add_count()")
                .isZero();
    }

    /**
     * Test close() - All public fields should be freed and set to null.
     */
    @Test
    public void closeResetsAllPublicMembers() {

        swigTrainData.swigLabelsChunkedArray.add(1);
        swigTrainData.swigFeaturesChunkedArray.add(1);
        swigTrainData.close();

        assertThat(swigTrainData.swigOutDatasetHandlePtr).as("swigOutDatasetHandlePtr").isNull();
        assertThat(swigTrainData.swigTrainLabelDataArray).as("swigTrainLabelDataArray").isNull();
        assertThat(swigTrainData.swigDatasetHandle).as("swigDatasetHandle").isNull();

        assertThat(swigTrainData.swigLabelsChunkedArray.get_chunks_count())
                .as("swigLabelsChunkedArray.get_chunks_count()")
                .isZero();
        assertThat(swigTrainData.swigLabelsChunkedArray.get_add_count())
                .as("swigLabelsChunkedArray.get_add_count()")
                .isZero();

        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunks_count())
                .as("swigFeaturesChunkedArray.get_chunks_count()")
                .isZero();
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_add_count())
                .as("swigFeaturesChunkedArray.get_add_count()")
                .isZero();
    }

}
