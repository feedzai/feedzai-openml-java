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
    private static final long NUM_FEATURES = 5;

    /**
     * Number of instances to store per chunk
     */
    private static final long NUM_INSTANCES_PER_CHUNK = 16;

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
        swigTrainData = new SWIGTrainData((int) NUM_FEATURES, NUM_INSTANCES_PER_CHUNK);
    }

    // TODO: test SWIGTrainData with fairnessConstrained=true

    /**
     * Assert the features ChunkedArray has the proper size.
     */
    @Test
    public void featuresChunkedArraySize() {
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunk_size())
                .as("features' ChunkedArray chunk size == NUM_FEATURES * NUM_INSTANCES_PER_CHUNK")
                .isEqualTo(NUM_FEATURES * NUM_INSTANCES_PER_CHUNK);
    }

    /**
     * The Features ChunkedArray size MUST be multiple of the instance size (numFeatures).
     * An instance cannot be broken across two partitions (chunks) when creating the
     * LightGBM Dataset.
     */
    @Test
    public void featuresChunkedArrayChunksHoldCompleteInstances() {
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunk_size() % NUM_FEATURES)
                .as("No instances split across feature chunks. chunkSize % numFeatures == 0")
                .isZero();
    }

    /**
     * Ensure features' chunks can hold the requested number of features.
     */
    @Test
    public void featuresChunkedArrayHoldsRequestedInstances() {
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunk_size() / NUM_FEATURES)
                .as("feature chunks hold requested number of instances")
                .isEqualTo(NUM_INSTANCES_PER_CHUNK);
    }

    /**
     * Not only features', but labels' ChunkedArray must hold chunks of the requested
     * `numInstancesPerChunk` size.
     */
    @Test
    public void labelChunkedArrayHoldsRequestedInstances() {
        assertThat(swigTrainData.swigLabelsChunkedArray.get_chunk_size())
                .as("label chunks hold requested number of instances")
                .isEqualTo(NUM_INSTANCES_PER_CHUNK);
    }

    /**
     * Test SWIGTrainResources() - some public members should be initialized.
     */
    @Test
    public void constructorInitializesPublicMembers() {
        assertThat(swigTrainData.swigOutDatasetHandlePtr).as("swigOutDatasetHandlePtr").isNotNull();

        assertThat(swigTrainData.swigLabelsChunkedArray.get_chunks_count())
                .as("starting count of label chunks")
                .isOne();
        assertThat(swigTrainData.swigLabelsChunkedArray.get_add_count())
                .as("starting add() count of label chunks")
                .isZero();

        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunks_count())
                .as("starting count of feature chunks")
                .isOne();
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_add_count())
                .as("starting add() count of feature chunks")
                .isZero();
    }

    /**
     * Test close() - All public fields should be freed and set to null.
     */
    @Test
    public void closeResetsAllPublicMembers() {

        // Simulate usage before close():
        swigTrainData.swigLabelsChunkedArray.add(1);
        swigTrainData.swigFeaturesChunkedArray.add(1);

        swigTrainData.close();

        assertThat(swigTrainData.swigOutDatasetHandlePtr).as("swigOutDatasetHandlePtr after close()").isNull();
        assertThat(swigTrainData.swigTrainLabelDataArray).as("swigTrainLabelDataArray after close()").isNull();
        assertThat(swigTrainData.swigDatasetHandle).as("swigDatasetHandle after close()").isNull();

        assertThat(swigTrainData.swigLabelsChunkedArray.get_chunks_count())
                .as("label chunks count after close()")
                .isZero();
        assertThat(swigTrainData.swigLabelsChunkedArray.get_add_count())
                .as("label chunks add() count after close()")
                .isZero();

        assertThat(swigTrainData.swigFeaturesChunkedArray.get_chunks_count())
                .as("feature chunks count after close()")
                .isZero();
        assertThat(swigTrainData.swigFeaturesChunkedArray.get_add_count())
                .as("feature chunks count after close()")
                .isZero();
    }

}
