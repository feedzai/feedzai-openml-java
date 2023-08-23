/*
 * Copyright 2022 Feedzai
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.feedzai.openml.provider.lightgbm;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.data.schema.NumericValueSchema;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import org.assertj.core.data.Offset;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.NUM_ITERATIONS_PARAMETER_NAME;
import static java.nio.file.Files.createTempFile;
import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;

/**
 * Test class to ensure that Microsoft LightGBM predicts the expected
 * <i>class distribution</i> and <i>feature contributions</i>.
 *
 * @author Sheng Wang (sheng.wang@feedzai.com)
 * @since 1.3.0
 */
public class LightGBMResultTest {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(LightGBMResultTest.class);

    /**
     * Parameters for model training.
     */
    private static final Map<String, String> MODEL_TRAINING_PARAMS = getTestParams();

    /**
     * Result Schema:
     * <ul>
     *     <li>card -> number</li>
     *     <li>amount -> number</li>
     *     <li>cat1_generator -> number</li>
     *     <li>cat2_generator -> number</li>
     *     <li>cat3_generator -> number</li>
     *     <li>num1_float -> number</li>
     *     <li>num2_float -> number</li>
     *     <li>num3_float -> number</li>
     *     <li>fraud_label -> number</li>
     * </ul>
     */
    public static final DatasetSchema SCHEMA = getTestSchema();

    /**
     * Result Schema:
     * <ul>
     *     <li>card -> number</li>
     *     <li>amount -> number</li>
     *     <li>cat1_generator -> number</li>
     *     <li>cat2_generator -> number</li>
     *     <li>cat3_generator -> number</li>
     *     <li>num1_float -> number</li>
     *     <li>num2_float -> number</li>
     *     <li>num3_float -> number</li>
     *     <li>bias -> number</li>
     *     <li>predictions -> number</li>
     * </ul>
     */
    public static final DatasetSchema RESULT_SCHEMA = getTestResultSchema();

    /**
     * The max number of instances to train, i.e., the number of lines in the file {@link treeshap_t treeshap_train.csv}.
     */
    public static final int MAX_INSTANCES_TO_TRAIN = 40000;

    /**
     * The chunk size instances, i.e., the number of chunk instances it keeps in memory.
     */
    public static final int CHUNK_SIZE_INSTANCES = 4;

    /**
     * Gets test schema.
     *
     * @return a {@link DatasetSchema}.
     */
    private static DatasetSchema getTestSchema() {
        final List<FieldSchema> schema = new ArrayList<>(9);
        addFields(schema);
        schema.add(new FieldSchema("fraud_label", 8, new NumericValueSchema(false)));

        return new DatasetSchema(
                8,
                schema
        );
    }

    /**
     * Gets test result schema.
     *
     * @return a {@link DatasetSchema}.
     */
    private static DatasetSchema getTestResultSchema() {
        final List<FieldSchema> schema = new ArrayList<>(10);
        addFields(schema);
        schema.add(new FieldSchema("bias", 8, new NumericValueSchema(false)));
        schema.add(new FieldSchema("predictions", 9, new NumericValueSchema(false)));

        return new DatasetSchema(
                0, // not used
                schema
        );
    }

    /**
     * Adds fields for both {@link #getTestSchema()} and {@link #getTestResultSchema()}.
     *
     * @param schema The {@link List} to add new fields.
     */
    private static void addFields(final List<FieldSchema> schema) {
        schema.add(new FieldSchema("card", 0, new NumericValueSchema(false)));
        schema.add(new FieldSchema("amount", 1, new NumericValueSchema(false)));
        schema.add(new FieldSchema("cat1_generator", 2, new NumericValueSchema(false)));
        schema.add(new FieldSchema("cat2_generator", 3, new NumericValueSchema(false)));
        schema.add(new FieldSchema("cat3_generator", 4, new NumericValueSchema(false)));
        schema.add(new FieldSchema("num1_float", 5, new NumericValueSchema(false)));
        schema.add(new FieldSchema("num2_float", 6, new NumericValueSchema(false)));
        schema.add(new FieldSchema("num3_float", 7, new NumericValueSchema(false)));
    }

    /**
     * Gets the parameters for LightGBM test.
     *
     * @return {@link Map} of key and values.
     */
    private static Map<String, String> getTestParams() {
        final Map<String, String> params = new HashMap<>(12, 1);
        params.put("max_bin", "512");
        params.put("learning_rate", "0.05");
        params.put("boosting_type", "gbdt");
        params.put("objective", "binary");
        params.put("num_leaves", "10");
        params.put("verbose", "-1");
        params.put("min_data", "100");
        params.put("boost_from_average", "True");
        params.put("seed", "42");
        params.put("num_iterations", "100");

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
     * Ensures that every train on LightGBM with the provided train dataset produces
     * the expected <i>class distribution</i> and <i>feature contributions</i>.
     *
     * @throws URISyntaxException    In case of error retrieving the data resource path.
     * @throws IOException           In case of error reading data.
     * @throws ModelLoadingException In case of error training the model.
     */
    @Test
    public void ensureLightgbmScoresAndContributions() throws URISyntaxException, IOException, ModelLoadingException {

        final Dataset dataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath("treeshap_t/treeshap_train.csv"),
                SCHEMA,
                MAX_INSTANCES_TO_TRAIN
        );
        final LightGBMBinaryClassificationModel model = fit(
                dataset,
                MODEL_TRAINING_PARAMS,
                CHUNK_SIZE_INSTANCES
        );
        final LightGBMTreeSHAPFeatureContributionExplainer explainer = new LightGBMTreeSHAPFeatureContributionExplainer(model);

        final ArrayList<List<Double>> classScores = new ArrayList<>(2);
        classScores.add(new LinkedList<>());
        classScores.add(new LinkedList<>());
        final ArrayList<List<Double>> classContributions = new ArrayList<>(2);
        classContributions.add(new LinkedList<>());
        classContributions.add(new LinkedList<>());

        final Iterator<Instance> iterator = dataset.getInstances();

        final Dataset resultDataset = CSVUtils.getDatasetWithSchema(
                TestResources.getResourcePath("treeshap_t/treeshap_result.csv"),
                RESULT_SCHEMA,
                MAX_INSTANCES_TO_TRAIN
        );

        final Iterator<Instance> resultIterator = resultDataset.getInstances();

        while (iterator.hasNext() && resultIterator.hasNext()) {
            final Instance instance = iterator.next();
            final Instance resultInstance = resultIterator.next();

            final double[] scoreDistribution = model.getClassDistribution(instance);
            final double[] featureContributions = explainer.getFeatureContributions(instance);

            assertThat(scoreDistribution.length)
                    .as("Class distribution should contain two values.")
                    .isEqualTo(2);

            final double prediction = resultInstance.getValue(9);
            final Offset<Double> offset = Offset.offset(1.0e-15);

            assertThat(scoreDistribution[1])
                    .as("The score should be expected.")
                    .isCloseTo(prediction, offset);

            for (int i = 0; i < featureContributions.length; i++) {
                assertThat(featureContributions[i])
                        .as("The contribution should be expected.")
                        .isCloseTo(resultInstance.getValue(i), offset);
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
