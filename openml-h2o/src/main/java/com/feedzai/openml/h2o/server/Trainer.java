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

package com.feedzai.openml.h2o.server;

import hex.Model;
import water.Job;

import java.io.IOException;

/**
 * A representation of an entity that knows how to train a specific {@link Job of the model}.
 *
 * @author Pedro Rijo (pedro.rijo@fedzai.com)
 * @since 0.1.0
 */
@FunctionalInterface
public interface Trainer <M extends Model> {

    /**
     * Trains the model.
     *
     * @return The {@link Job resulting trained model}.
     * @throws IOException If any problem occurs connecting to the H2O server.
     */
    Job<M> train() throws IOException;
}
