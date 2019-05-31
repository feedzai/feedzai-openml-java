package com.feedzai.openml.h2o;

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.data.schema.StringValueSchema;
import com.feedzai.openml.mocks.MockInstance;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.assertj.core.api.Assertions;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Regression test for PULSEDEV-27879 to ensure we can load a H2O model with String feature fields.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 1.0.5
 */
public class ImportAllFeaturesTypesTest {

    /**
     * Tests that H2O can load a model that was trained with a string feature.
     *
     * @throws ModelLoadingException If any unexpected error occurs.
     */
    @Test
    public void test() throws ModelLoadingException {

        final DatasetSchema schema = new DatasetSchema(4, ImmutableList.of(
                new FieldSchema("Amount", 0, new NumericValueSchema(false)),
                new FieldSchema("MerchantID", 1, new StringValueSchema(true)),
                new FieldSchema("MCC", 2, new CategoricalValueSchema(true, ImmutableSet.of("1711", "3010", "5411", "5812", "6011", "7995", "8398"))),
                new FieldSchema("AccountCreatedAt", 3, new NumericValueSchema(true)),
                new FieldSchema("FraudLabel", 4, new CategoricalValueSchema(true, ImmutableSet.of("GENUINE", "FRAUD")))
        ));

        final String modelPath = this.getClass().getResource("/drf-all-types-features").getPath();
        final AbstractClassificationH2OModel model = new H2OModelCreator(H2OAlgorithm.DISTRIBUTED_RANDOM_FOREST.getAlgorithmDescriptor())
                .loadModel(Paths.get(modelPath), schema);

        final Instance instance = new MockInstance(ImmutableList.of(14923.103, "Amazon", 0.0, 1559312575000.0, 0.0));

        assertThat(model.classify(instance))
                .as("the score")
                .isBetween(0, 1);

        assertThat(Arrays.stream(model.getClassDistribution(instance)).sum())
                .as("sum of the class distribution", new Object[0])
                .isCloseTo(1.0D, Assertions.within(0.01D));
    }


}
