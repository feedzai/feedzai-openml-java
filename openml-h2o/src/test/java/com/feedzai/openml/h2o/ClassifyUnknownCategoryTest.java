package com.feedzai.openml.h2o;

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.exception.ModelTrainingException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.assertj.core.api.Assertions;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Regression test for PULSEDEV-24998.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 0.1.0
 */
public class ClassifyUnknownCategoryTest {

    /**
     * Tests that H2O can score an instance containing a categorical value that was not present in the training data.
     *
     * @throws ModelTrainingException If any unexpected error occurs.
     * @since 0.1.0
     */
    @Test
    public void test() throws ModelTrainingException {

        final DatasetSchema schema = new DatasetSchema(0, ImmutableList.of(
                new FieldSchema("isFraud", 0, new CategoricalValueSchema(true, ImmutableSet.of("0", "1"))),
                new FieldSchema("catFeature", 1, new CategoricalValueSchema(true, ImmutableSet.of("A", "B")))
        ));
        final MockDataset dataset = new MockDataset(schema, ImmutableList.of(
                new MockInstance(ImmutableList.of(0.0, 0.0)),
                new MockInstance(ImmutableList.of(1.0, 0.0))
        ));
        final AbstractClassificationH2OModel model = new H2OModelCreator(H2OAlgorithm.XG_BOOST.getAlgorithmDescriptor())
                .fit(dataset, new Random(0), H2OAlgorithmTestParams.getXgboost());

        final int score = model.classify(
                new MockInstance(ImmutableList.of(1.0, 1.0))
        );
        assertThat(score)
                .as("the score")
                .isBetween(0, 1);

        final double[] scoresDist = model.getClassDistribution(
                new MockInstance(ImmutableList.of(0.0, 1.0))
        );
        assertThat(Arrays.stream(scoresDist).sum())
                .as("sum of the class distribution", new Object[0])
                .isCloseTo(1.0D, Assertions.within(0.01D));
    }


}
