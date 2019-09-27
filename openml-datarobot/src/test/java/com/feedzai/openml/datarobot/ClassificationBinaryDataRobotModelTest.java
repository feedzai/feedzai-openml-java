package com.feedzai.openml.datarobot;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.google.common.primitives.Doubles;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for classifying DataRobot Model with {@link ClassificationBinaryDataRobotModel}.
 *
 * @author Sheng Wang (sheng.wang@feedzai.com)
 * @since 1.0.6
 */
public class ClassificationBinaryDataRobotModelTest extends DataRobotModelProviderLoadTest {

    /**
     * Checks the method ClassificationBinaryDataRobotModel#classify() if it ensures
     * that DataRobot classifies correctly the index of maximum value of the scores' list.
     *
     * @throws ModelLoadingException If the model cannot be loaded.
     *
     * @since 1.0.6
     */
    @Test
    public final void testClassify() throws ModelLoadingException {
        final String modelPath = this.getClass().getResource("/boolean_classifier").getPath();
        final DataRobotModelCreator loader = getMachineLearningModelLoader(getValidAlgorithm());
        final DatasetSchema schema = loader.loadSchema(Paths.get(modelPath));
        final ClassificationBinaryDataRobotModel model = loader.loadModel(Paths.get(modelPath), schema);

        MockInstance instance = new MockInstance(schema, new Random());
        final double[] scores = model.getClassDistribution(instance);
        final int classificationIndex = model.classify(instance);
        final double maxScore = Arrays.stream(scores).max().getAsDouble();

        assertThat(Doubles.asList(scores).indexOf(maxScore))
                .as("The index of maximum value")
                .isEqualTo(classificationIndex);
    }

}