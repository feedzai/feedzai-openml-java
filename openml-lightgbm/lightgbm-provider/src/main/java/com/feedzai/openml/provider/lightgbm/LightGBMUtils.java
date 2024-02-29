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
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.google.common.io.ByteStreams.copy;

/**
 * Collection of util methods specific to the LightGBM OpenML Provider.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.0.10
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
    static final int BINARY_LGBM_NUM_CLASSES = 1;

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
            final CpuArchitecture cpuArchitecture = getCpuArchitecture(System.getProperty("os.arch"));
            final boolean isAlpine = isAlpine();

            try {
                loadSharedLibraryFromJar("libgomp.so.1.0.0", cpuArchitecture, isAlpine);
                loadSharedLibraryFromJar("lib_lightgbm.so", cpuArchitecture, isAlpine);
                loadSharedLibraryFromJar("lib_lightgbm_swig.so", cpuArchitecture, isAlpine);
            } catch (final IOException ex) {
                throw new RuntimeException("Failed to load LightGBM shared libraries from jar.", ex);
            }

            logger.info("Loaded LightGBM libs.");
            libsLoaded = true;
        }
    }

    static CpuArchitecture getCpuArchitecture(final String cpuArchName) {
        try {
            return CpuArchitecture.valueOf(cpuArchName.toUpperCase());
        } catch (final IllegalArgumentException ex) {
            logger.error("Trying to use LightGBM on an unsupported architecture {}.", cpuArchName, ex);
            throw ex;
        }
    }

    /**
     * Checks if the current operative system is Alpine.
     *
     * @return true when the operative system is Alpine and false otherwise.
     */
    static private boolean isAlpine() {
        final String osReleasePath = "/etc/os-release";

        try {
            BufferedReader reader = new BufferedReader(new FileReader(osReleasePath));

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.equals("ID=alpine")) {
                    reader.close();
                    return true;
                }
            }
        } catch (IOException ex) {
            return false;
        }

        return false;
    }

    /**
     * Loads a single shared library from the Jar.
     *
     * @param sharedLibResourceName library "filename" inside the jar.
     * @param cpuArchitecture cpu architecture.
     * @throws IOException if any error happens loading the library.
     */
    static private void loadSharedLibraryFromJar(
            final String sharedLibResourceName,
            final CpuArchitecture cpuArchitecture,
            final boolean isAlpine
    ) throws IOException {

        if (isAlpine && cpuArchitecture.getLgbmNativeLibsFolder().equals("arm64")) {
            throw new IOException("Trying to use LightGBM on an unsupported architecture arm64 on Alpine.");
        }

        final String libraryPath;
        if (cpuArchitecture.getLgbmNativeLibsFolder().equals("amd64") && isAlpine) {
            logger.debug("Loading LightGBM shared lib: {} for {} on Alpine.", sharedLibResourceName, cpuArchitecture);
            libraryPath = cpuArchitecture.getLgbmNativeLibsFolder() + "/alpine/" + sharedLibResourceName;
        }
        else if (cpuArchitecture.getLgbmNativeLibsFolder().equals("amd64")) {
            logger.debug("Loading LightGBM shared lib: {} for {}.", sharedLibResourceName, cpuArchitecture);
            libraryPath = cpuArchitecture.getLgbmNativeLibsFolder() + "/glibc/" + sharedLibResourceName;
        }
        else {
            logger.debug("Loading LightGBM shared lib: {} for {}.", sharedLibResourceName, cpuArchitecture);
            libraryPath = cpuArchitecture.getLgbmNativeLibsFolder() + "/" + sharedLibResourceName;
        }

        final InputStream inputStream = LightGBMUtils.class.getClassLoader()
                .getResourceAsStream(libraryPath);
        final File tempFile = File.createTempFile("lib", ".so");
        final OutputStream outputStream = new FileOutputStream(tempFile);

        copy(inputStream, outputStream);
        inputStream.close();
        outputStream.close();

        System.load(tempFile.getAbsolutePath());
        tempFile.deleteOnExit();
    }
}
