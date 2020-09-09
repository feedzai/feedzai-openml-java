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

import com.feedzai.openml.provider.exception.ModelLoadingException;
import org.junit.BeforeClass;
import org.junit.Test;

import java.net.URISyntaxException;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;

/**
 * Tests for the SWIGResources class to ensure all resources are properly/safely initialized and released.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.18
 */
public class SWIGResourcesTest {

    /**
     * Standard model path.
     */
    static Path modelPath;

    /**
     * SWIGResources instance common to all tests. Do not modify it!
     * Used to speed up tests by avoiding re-reading the model file.
     */
    static SWIGResources defaultSwig;

    /**
     * Set up the standard model path.
     *
     * @throws URISyntaxException Couldn't extract file resource.
     */
    @BeforeClass
    static public void setupFixture() throws ModelLoadingException, URISyntaxException {

        LightGBMUtils.loadLibs(); // Initialize LightGBM libs.

        modelPath = TestResources.getModelFilePath();
        defaultSwig = new SWIGResources(modelPath.toString(), "");
    }

    /**
     * Test SWIGResources() - all public members should be initialized.
     */
    @Test
    public void constructorInitializesAllPublicMembers() {

        assertThat(defaultSwig.swigBoosterHandle).as("swigBoosterHandle").isNotNull();
        assertThat(defaultSwig.swigFastConfigHandle).as("swigFastConfigHandle").isNotNull();
        assertThat(defaultSwig.swigInstancePtr).as("swigInstancePtr").isNotNull();
        assertThat(defaultSwig.swigOutIntPtr).as("swigOutIntPtr").isNotNull();
        assertThat(defaultSwig.swigOutLengthInt64Ptr).as("swigOutLengthInt64Ptr").isNotNull();
        assertThat(defaultSwig.swigOutScoresPtr).as("swigOutScoresPtr").isNotNull();
    }

    /**
     * Test SWIGResources() - A ModelLoadingException is thrown when the modelPath does not point to a valid model file.
     */
    @Test
    public void constructorThrowsModelLoadingExceptionOnInvalidModelPath() {

        assertThatExceptionOfType(ModelLoadingException.class).isThrownBy(() ->
                new SWIGResources("__invalid_model_path__", ""));
    }

    /**
     * Test close() - All public fields should be freed and set to null.
     */
    @Test
    public void closeResetsAllPublicMembers() throws ModelLoadingException {

        // Generate a new SWIGResources instance as it will be modified:
        SWIGResources swig = new SWIGResources(modelPath.toString(), "");
        swig.close();

        assertThat(swig.swigBoosterHandle).as("swigBoosterHandle").isNull();
        assertThat(swig.swigFastConfigHandle).as("swigFastConfigHandle").isNull();
        assertThat(swig.swigInstancePtr).as("swigInstancePtr").isNull();
        assertThat(swig.swigOutIntPtr).as("swigOutIntPtr").isNull();
        assertThat(swig.swigOutLengthInt64Ptr).as("swigOutLengthInt64Ptr").isNull();
        assertThat(swig.swigOutScoresPtr).as("swigOutScoresPtr").isNull();
    }

    /**
     * Test getBoosterNumIterations() - Should return the correct number.
     */
    @Test
    public void getBoosterNumIterationsReturnsCorrectValue() {

        assertThat(defaultSwig.getBoosterNumIterations()).as("number of booster iterations").isEqualTo(200);
    }

    /**
     * Test getBoosterNumFeatures() - Should return the correct number.
     */
    @Test
    public void getBoosterNumFeaturesReturnsCorrectValue() {

        assertThat(defaultSwig.getBoosterNumFeatures()).as("number of booster features").isEqualTo(4);
    }
}
