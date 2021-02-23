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

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for the SWIGTrainResources class to ensure all resources are properly/safely initialized and released.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.1.2 (renamed from SWIGTrainResourcesTest)
 */
public class SWIGTrainBoosterTest {

    /**
     * SWIGTrainResources instance common to all tests.
     * Fresh instance initialized before each test.
     */
    private SWIGTrainBooster swigTrainBooster;

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
        swigTrainBooster = new SWIGTrainBooster();
    }

    /**
     * Test SWIGTrainResources() - some public members should be initialized.
     */
    @Test
    public void constructorInitializesPublicMembers() {
        assertThat(swigTrainBooster.swigOutBoosterHandlePtr).as("swigOutBoosterHandlePtr").isNotNull();
    }

    /**
     * Test close() - All public fields should be freed and set to null.
     */
    @Test
    public void closeResetsAllPublicMembers() {

        swigTrainBooster.close();

        assertThat(swigTrainBooster.swigOutBoosterHandlePtr).as("swigOutBoosterHandlePtr").isNull();
        assertThat(swigTrainBooster.swigBoosterHandle).as("swigBoosterHandle").isNull();
    }

}
