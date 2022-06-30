/*
 * Copyright 2018 Feedzai
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
 *
 */

package com.feedzai.openml.h2o;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.util.data.encoding.EncodingHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.supercsv.io.CsvListWriter;
import org.supercsv.prefs.CsvPreference;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

/**
 * Utility class for H2O related tasks.
 *
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 0.1.0
 */
public final class H2OUtils {

    /**
     * Logger for {@link H2OUtils}.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OUtils.class);

    /**
     * Prefix for files created during the H2O model training.
     */
    public static final String FEEDZAI_H2O_PREFIX = "feedzai-h2o-";

    /**
     * Private constructor to protect this class from instantiated.
     */
    private H2OUtils() { }

    /**
     * Writes a {@link Dataset} to disk in a CSV format to be fed into the H2O platform.
     *
     * @param dataset The dataset to write.
     * @return The {@link Path} where the dataset was stored.
     * @throws IOException If a problem occurs writing to disk.
     */
    public static Path writeDatasetToDisk(final Dataset dataset) throws IOException {

        final Path datasetPath = Files.createTempFile(FEEDZAI_H2O_PREFIX + UUID.randomUUID(), ".dataset");
        logger.info("Writing dataset to disk: {}", datasetPath);

        final DatasetSchema schema = dataset.getSchema();

        try (final CsvListWriter csvWriter = new CsvListWriter(
                new FileWriter(datasetPath.toFile()),
                new CsvPreference.Builder('"', ',', "\r\n").build()
        )) {

            StreamSupport.stream(((Iterable<Instance>) dataset::getInstances).spliterator(), false)
                    .map(instance -> IntStream.range(0, schema.getFieldSchemas().size())
                            .boxed()
                            .map(featureIdx -> rawValue(instance, schema, featureIdx))
                            .collect(Collectors.toList()))
                    .forEach(row -> {
                        try {
                            csvWriter.write(row);
                        } catch (final IOException e) {
                            logger.error(String.format("Error writing dataset as csv. Row: %s", row), e);
                        }
                    });
        }

        return datasetPath;
    }

    /**
     * Extract raw feature value of an instance. Since some type of features
     * (for instance {@link CategoricalValueSchema Categorical features}) may be encoded, this method extracts the
     * original value.
     *
     * @param instance   The instance holding the value.
     * @param schema     The schema representation.
     * @param featureIdx The index of the feature.
     * @return A raw representation of the feature value, as a String.
     */
    private static String rawValue(final Instance instance, final DatasetSchema schema, final int featureIdx) {
        final List<FieldSchema> fields = schema.getFieldSchemas();

        final double featureValue = instance.getValue(featureIdx);
        final FieldSchema featureSchema = fields.get(featureIdx);

        if (featureSchema.getValueSchema() instanceof CategoricalValueSchema) {
            return EncodingHelper.decodeDoubleToCategory(featureValue, (CategoricalValueSchema) featureSchema.getValueSchema());
        } else {
            return Double.toString(featureValue);
        }
    }

}
