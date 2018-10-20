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

package com.feedzai.openml.h2o.server.export;

import com.feedzai.openml.provider.exception.ModelTrainingException;
import hex.Model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * An abstract representation of an H2O exported model.
 *
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 * @since 0.1.0
 */
public interface ExportedModel {

    /**
     * Writes the exported model to disk, into the specified directory.
     *
     * @param exportDir The directory where to save the model.
     * @param model     The model.
     * @throws IOException If any problem occurs accessing the file system.
     */
    void save(final Path exportDir, final Model model) throws IOException, ModelTrainingException;
}
