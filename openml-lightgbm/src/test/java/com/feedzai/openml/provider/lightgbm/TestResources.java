/*
 * Copyright 2020 Feedzai
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

package com.feedzai.openml.provider.lightgbm;

import org.apache.commons.csv.CSVParser;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * This serves as a configuration class to manage the test resources.
 * It has all the reference resources necessary to perform the tests.
 *
 * @author Alberto Ferreira
 * @since 1.0.10
 */
public class TestResources {

    /**
     * Name of the scored instances resource.
     */
    public static final String SCORED_INSTANCES_NAME = "scored_instances.csv";

    /**
     * Model folder resource name.
     */
    public static final String MODEL_FOLDER_RESOURCE_NAME = "example_model_folder/";

    /**
     * Model file resource name.
     */
    public static final String MODEL_FILE_RESOURCE_NAME = MODEL_FOLDER_RESOURCE_NAME + "LightGBM_model.txt";

    /**
     * Returns a resource path from a resource in the tests.
     *
     * @param resourceName Name of the resource to get the path from.
     * @return resource path
     * @throws URISyntaxException if extracting that resource was not possible.
     */
    public static Path getResourcePath(final String resourceName) throws URISyntaxException {

        return Paths.get(TestResources.class.getClassLoader().getResource(resourceName).toURI());
    }

    /**
     * Gets model file path.
     *
     * @return Path of test model resource.
     * @throws URISyntaxException if extracting that resource was not possible.
     */
    public static Path getModelFilePath() throws URISyntaxException {

        return getResourcePath(MODEL_FILE_RESOURCE_NAME);
    }

    /**
     * Gets model folder path.
     *
     * @return Path to the model folder resource.
     * @throws URISyntaxException if extracting that resource was not possible.
     */
    public static Path getModelFolderPath() throws URISyntaxException {

        return getResourcePath(MODEL_FOLDER_RESOURCE_NAME);
    }

    /**
     * Gets scored instances path.
     *
     * @return path of the scored instances
     * @throws URISyntaxException if extracting that resource was not possible.
     */
    public static Path getScoredInstancesPath() throws URISyntaxException {

        return getResourcePath(SCORED_INSTANCES_NAME);
    }

    /**
     * Gets scored instances CSV parser.
     *
     * @return CSVParser for the scored instances.
     * @throws URISyntaxException in case the resource URI cannot be created.
     * @throws IOException        in case there's an IO error.
     */
    public static CSVParser getScoredInstancesCSVParser() throws URISyntaxException, IOException {

        return CSVUtils.getCSVParser(getScoredInstancesPath());
    }

}
