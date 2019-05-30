/*
 * Copyright 2019 Feedzai
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

import com.feedzai.openml.java.utils.AbstractWideTypesSupportTest;
import com.feedzai.openml.provider.model.MachineLearningModelLoader;

import static com.feedzai.openml.datarobot.AbstractDataRobotModelProviderLoadTest.SECOND_MODEL_FILE;

/**
 * Regression for PULSEDEV-27879 for DataRobot.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 1.0.5
 */
public class DataRobotWideTypesSupportTest extends AbstractWideTypesSupportTest {

    @Override
    protected String getModelPath() {
        return this.getClass().getResource("/" + SECOND_MODEL_FILE).getPath();
    }

    @Override
    protected MachineLearningModelLoader<?> getModelLoader() {
        return new DataRobotModelCreator();
    }

}
