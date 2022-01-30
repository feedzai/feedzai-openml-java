/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * (c) 2022 Feedzai, Strictly Confidential
 */

package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static java.lang.String.format;
import static java.nio.file.Files.createTempFile;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * FIXME
 *
 * @author Sheng Wang (sheng.wang@feedzai.com)
 * @since 1.2.2
 */
public class LightGBMResultTest {

    /**
     * Parameters for model train.
     */
    private static final Map<String, String> MODEL_PARAMS = getTestParams();


    /**
     * Schema:
     * card -> int
     * amount -> int
     * cat1_generator  -> int
     * cat2_generator -> int
     * cat3_generator -> int
     * num1_float -> float
     * num2_float -> float
     * num3_float -> float
     */
    public static final DatasetSchema SCHEMA = getTestSchema();

    public static final int MAX_INSTANCES_TO_TRAIN = 100;

    public static final int CHUNK_SIZE_INSTANCES = 10;

    private static DatasetSchema getTestSchema() {
        final List<FieldSchema> schema = new ArrayList<>(8);
        schema.add(new FieldSchema("card", 0, new NumericValueSchema(false)));
        schema.add(new FieldSchema("amount", 1, new NumericValueSchema(false)));
        schema.add(new FieldSchema("cat1_generator", 2, new NumericValueSchema(false)));
        schema.add(new FieldSchema("cat2_generator", 3, new NumericValueSchema(false)));
        schema.add(new FieldSchema("cat3_generator", 4, new NumericValueSchema(false)));
        schema.add(new FieldSchema("num1_float", 5, new NumericValueSchema(false)));
        schema.add(new FieldSchema("num2_float", 6, new NumericValueSchema(false)));
        schema.add(new FieldSchema("num3_float", 7, new NumericValueSchema(false)));

        return new DatasetSchema(
                0,
                schema
        );
    }

    private static Map<String, String> getTestParams() {
        final Map<String, String> params = new HashMap<>(12, 1);
        params.put("max_bin", "512");
        params.put("learning_rate", "0.05");
        params.put("boosting_type", "gbdt");
        params.put("objective", "binary");
        params.put("metric", "binary_logloss");
        params.put("num_leaves", "10");
        params.put("verbose", "-1");
        params.put("min_data", "100");
        params.put("boost_from_average", "True");
        params.put("seed", "42");
        params.put("num_iterations", "100");

        params.put("early_stopping_rounds", "50");
        params.put("num_boost_round", "1000");
        return params;
    }

    /**
     * Load the LightGBM utils or nothing will work.
     * Also changes parameters.
     */
    @BeforeClass
    public static void setupFixture() {
        LightGBMUtils.loadLibs();
    }


    /**
     * Asserts that a model trained with numericals only and evaluated on the same datasource
     * has in average higher scores for the positive class (1) than for the negative one (0).
     *
     * @throws URISyntaxException    In case of error retrieving the data resource path.
     * @throws IOException           In case of error reading data.
     * @throws ModelLoadingException In case of error training the model.
     */
    @Test
    public void trainLightgbm() throws URISyntaxException, IOException, ModelLoadingException {

        final Dataset dataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath("treeshap_t/treeshap_train.csv"),
                SCHEMA,
                MAX_INSTANCES_TO_TRAIN
        );
        final LightGBMBinaryClassificationModel model = fit(
                dataset,
                MODEL_PARAMS,
                CHUNK_SIZE_INSTANCES
        );

        final int targetIndex = dataset.getSchema().getTargetIndex().get();

        final ArrayList<List<Double>> classScores = new ArrayList<>(2);
        classScores.add(new LinkedList<>());
        classScores.add(new LinkedList<>());
        final ArrayList<List<Double>> classContributions = new ArrayList<>(2);
        classContributions.add(new LinkedList<>());
        classContributions.add(new LinkedList<>());

        final Iterator<Instance> iterator = dataset.getInstances();

        final int maxInstances = 100;   // FIXME
        for (int numInstances = 0; iterator.hasNext() && numInstances < maxInstances; ++numInstances) {
            final Instance instance = iterator.next();
            // final int classIndex = (int) instance.getValue(targetIndex);
            // get scores and contributions
            final double[] scoreDistribution = model.getClassDistribution(instance);
            final double[] featureContributions = model.getFeatureContributions(instance);

            for (int i = 0; i < scoreDistribution.length; i++) {
                System.out.println(format("class index [%d] with distribution [%d]", i, scoreDistribution[i]));
            }
            for (int i = 0; i < featureContributions.length; i++) {
                System.out.println(format("class index [%d] with contribution [%d]", i, featureContributions[i]));
            }
        }
    }

    /**
     * Mimics LightGBMModelCreator#fit but allows customizing instancesPerChunk.
     *
     * @param dataset           train dataset.
     * @param params            LightGBM params.
     * @param instancesPerChunk tune memory layout for train data buffer.
     * @return trained LightGBM model.
     * @throws IOException           Can be thrown when creating a temporary model file.
     * @throws ModelLoadingException If there was any issue training the model.
     */
    private static LightGBMBinaryClassificationModel fit(
            final Dataset dataset,
            final Map<String, String> params,
            final long instancesPerChunk) throws IOException, ModelLoadingException {

        final Path tmpModelFilePath = createTempFile("pulse_lightgbm_model_", null);

        try {
            LightGBMBinaryClassificationModelTrainer.fit(
                    dataset, params, tmpModelFilePath, instancesPerChunk);
            return new LightGBMModelCreator().loadModel(tmpModelFilePath, dataset.getSchema());
        } finally {
            Files.delete(tmpModelFilePath);
        }
    }
}
