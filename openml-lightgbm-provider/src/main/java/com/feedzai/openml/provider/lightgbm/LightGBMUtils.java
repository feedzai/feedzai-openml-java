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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

import static com.google.common.io.ByteStreams.copy;

/**
 * Collection of util methods specific to the LightGBM OpenML Provider.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 */
public class LightGBMUtils {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(LightGBMUtils.class);

    /**
     * LightGBM treats binary classification as a special case of 1 class.
     * I.e.: For binary classification it outputs a single value.
     */
    public static final int BINARY_LGBM_NUM_CLASSES = 1;

    /**
     * State variable to know if it loadLibs was ever called.
     */
    private static boolean libsLoaded = false;

    /**
     * Loads the SWIG LightGBM dynamic libraries.
     * Repeated calls are ignored by System.loadLibrary.
     * WARNING: Before calling this method, no lightgbmlib* calls are allowed!
     */
    static public void loadLibs() {

        if (!libsLoaded) {
            try {
                loadSharedLibraryFromJar("libgomp.so.1.0.0");
                loadSharedLibraryFromJar("lib_lightgbm.so");
                loadSharedLibraryFromJar("lib_lightgbm_swig.so");
            } catch (final IOException e) {
                throw new RuntimeException("Failed to load LightGBM shared libraries from jar.", e);
            }

            logger.info("Loaded LightGBM libs.");
            libsLoaded = true;
        }
    }

    /**
     * Loads a single shared library from the Jar.
     *
     * @param sharedLibResourceName library "filename" inside the jar.
     * @throws IOException if any error happens loading the library.
     */
    static private void loadSharedLibraryFromJar(final String sharedLibResourceName) throws IOException {

        logger.debug("Loading LightGBM shared lib: {}.", sharedLibResourceName);

        final InputStream inputStream = LightGBMUtils.class.getClassLoader().getResourceAsStream(sharedLibResourceName);
        final File tempFile = File.createTempFile("lib", ".so");
        final OutputStream outputStream = new FileOutputStream(tempFile);

        copy(inputStream, outputStream);
        inputStream.close();
        outputStream.close();

        System.load(tempFile.getAbsolutePath());
        tempFile.deleteOnExit();
    }
}
