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

package com.feedzai.openml.datarobot;

import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.SortedSet;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for loading models with {@link DataRobotModelProvider}.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public class DataRobotModelProviderLoadTest extends AbstractDataRobotModelProviderLoadTest {

    /**
     * Checks that is possible to use compatible target values with the values used to train the model.
     *
     * @throws ModelLoadingException If the target are incompatible.
     */
    @Test
    public void compatibleTargetValuesWithSchema() throws ModelLoadingException {
        final Set<String> nominalValues = getFirstModelTargetNominalValues();
        final String[] targetModelValues = nominalValues.toArray(new String[]{});
        final SortedSet<String> sortedNominalValues = getFirstMachineLearningModelLoader()
                .checkTargetModelValuesWithSchema(
                        createDatasetSchema(nominalValues),
                        targetModelValues
                );

        assertThat(sortedNominalValues)
                .as("the target values used to train the model")
                .containsExactlyInAnyOrder(targetModelValues);
    }

    /**
     * Checks that is not possible to use incompatible target values with the values used to train the model.
     *
     * @throws ModelLoadingException If the target are incompatible.
     */
    @Test(expected = ModelLoadingException.class)
    public void incompatibleTargetValuesWithSchema() throws ModelLoadingException {
        final Set<String> nominalValues = getFirstModelTargetNominalValues();
        final String[] targetModelValues = new String[]{"true", "false"};
        getFirstMachineLearningModelLoader().checkTargetModelValuesWithSchema(
                createDatasetSchema(nominalValues),
                targetModelValues
        );
    }

    /**
     * Checks that is possible to use binary target values on DataRobot models.
     */
    @Test
    public void targetIsBinaryTest() {
        final List<ParamValidationError> validationErrors = getFirstMachineLearningModelLoader()
                .validateTargetIsBinary(createDatasetSchema(getFirstModelTargetNominalValues()));

        assertThat(validationErrors)
                .as("list of errors")
                .isEmpty();
    }

    /**
     * Checks that is not possible to use non-binary target values on DataRobot models.
     */
    @Test
    public void targetIsNotBinaryTest() {
        final MockDataset mockDataset = new MockDataset(
                ImmutableSet.of("c1", "c2", "c3"),
                10,
                1,
                new Random(123)
        );

        final List<ParamValidationError> validationErrors = getFirstMachineLearningModelLoader()
                .validateTargetIsBinary(mockDataset.getSchema());

        assertThat(validationErrors)
                .as("list of errors")
                .hasSize(1);
    }

    /**
     * Checks that is possible to use Jar files to import DataRobot models..
     */
    @Test
    public void validModelFileFormatTest() {
        final Path modelPath = Paths.get(getClass().getResource("/" + getValidModelDirName()).getPath());
        final List<ParamValidationError> validationErrors = getFirstMachineLearningModelLoader()
                .validateModelFileFormat(modelPath);

        assertThat(validationErrors)
                .as("list of errors")
                .isEmpty();
    }

    /**
     * Checks that is not possible to use invalid file formats to import DataRobot models..
     */
    @Test
    public void invalidModelFileFormatTest() throws IOException {
        final File tempFile = File.createTempFile("temp-file-name", ".tmp");
        tempFile.deleteOnExit();

        final List<ParamValidationError> validationErrors = getFirstMachineLearningModelLoader()
                .validateModelFileFormat(tempFile.toPath());

        assertThat(validationErrors)
                .as("list of errors")
                .hasSize(1);
    }
}
