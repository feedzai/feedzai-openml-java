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
import hex.ModelMojoWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.zip.ZipOutputStream;

/**
 * Concrete implementation of an {@link ExportedModel} that represents an {@link hex.genmodel.MojoModel H2O MOJO}.
 *
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 * @since 0.1.0
 */
public class MojoExported implements ExportedModel {

    /**
     * Extension of files that contain a model stored in MOJO (Model Object, Optimized) format.
     */
    public static final String MOJO_EXTENSION = "zip";

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(MojoExported.class);

    /**
     * The zip directory for the created MOJO file. This string is the relative path were the directory
     * with the model will be stored inside the compressed file, so now this directory
     * is being stored in a root directory of the compressed file.
     */
    private static final String ZIP_DIRECTORY = "";

    /**
     * Creates a new instance of this class.
     */
    public MojoExported() {
    }

    @Override
    public void save(final Path exportDir, final Model model) throws ModelTrainingException {
        final String modelFilename = model._output._job._result.toString();

        logger.info("Saving exported MOJO into " + modelFilename);

        try (final FileOutputStream fos = new FileOutputStream(String.format("%s%c%s%c%s", exportDir, File.separatorChar, modelFilename, '.', MOJO_EXTENSION));
             final BufferedOutputStream bos = new BufferedOutputStream(fos);
             final ZipOutputStream zos = new ZipOutputStream(bos)) {

            final ModelMojoWriter mojo = model.getMojo();
            mojo.writeTo(zos, ZIP_DIRECTORY);

        } catch (final IOException e) {
            final String errorMsg = "Unable to store the model on a MOJO " + modelFilename;
            logger.error(errorMsg, e);
            throw new ModelTrainingException(errorMsg, e);
        }
    }
}
