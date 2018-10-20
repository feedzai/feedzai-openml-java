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

package com.feedzai.openml.java.utils;

import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.google.common.io.Files;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Objects;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

/**
 * Contains the common utility methods used by OpenML providers to interact with Java files.
 *
 * @author Paulo Pereira (paulo.pereira@feedzai.com)
 * @since 0.1.0
 */
public class JavaFileUtils {

    /**
     * Constructor of the utility class.
     */
    private JavaFileUtils() {
    }

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(JavaFileUtils.class);

    /**
     * Extension of jar files.
     */
    public static final String JAR_EXTENSION = "jar";

    /**
     * Extension of Java files.
     */
    public static final String JAVA_FILE_EXTENSION = "java";

    /**
     * Extension of compiled Java files.
     */
    private static final String JAVA_COMPILED_EXTENSION = "class";

    /**
     * Gets a {@link URLClassLoader} used to retrieve an instance of class that contains the model.
     *
     * @param modelPath Path of the binary with the model.
     * @return a {@link URLClassLoader} used to retrieve an instanece of the model.
     * @throws ModelLoadingException If it cannot get the path of the model.
     */
    public static URLClassLoader getUrlClassLoader(final String modelPath,
                                                   final ClassLoader classLoader) throws ModelLoadingException {
        final URL resourceURL;
        try {
            resourceURL = Paths.get(modelPath).toUri().toURL();
        } catch (final MalformedURLException e) {
            logger.error("Could not get the path of the model [{}].", modelPath, e);
            throw new ModelLoadingException(
                    String.format("An error was found when getting the path of the model [%s]", modelPath),
                    e
            );
        }
        return URLClassLoader.newInstance(new URL[]{resourceURL}, classLoader);
    }

    /**
     * Loads a class from a {@link URLClassLoader} and creates new instance of that class.
     *
     * @param modelPath        Path of the binary with the model.
     * @param simpleNameFormat Format of the class/package of the instance to create.
     * @param urlClassLoader   The {@link URLClassLoader} used to retrieve an instance of the model.
     * @return The new instance of the loaded class.
     * @throws ModelLoadingException If it cannot create the instance.
     */
    public static Object createNewInstanceFromClassLoader(final String modelPath,
                                                          final String simpleNameFormat,
                                                          final URLClassLoader urlClassLoader) throws ModelLoadingException {
        try {
            final String simpleNameJar = FilenameUtils.getBaseName(modelPath);
            return urlClassLoader.loadClass(String.format(simpleNameFormat, simpleNameJar)).newInstance();
        } catch (final Exception e) {
            logger.error("Could not load the model [{}].", modelPath, e);
            throw new ModelLoadingException(
                    String.format("An error was found during the import of the model [%s]", modelPath),
                    e
            );
        }
    }

    /**
     * Verifies that the {@code filePath} received by parameter is equals to the {@link #JAR_EXTENSION extension of
     * a Jar file}.
     *
     * @param filePath The path of the file to be verified.
     * @return True if it has the Jar file, false otherwise.
     */
    static public boolean isJarFile(final String filePath) {
        return JAR_EXTENSION.equals(Files.getFileExtension(filePath));
    }

    /**
     * Initializes the compiler using the ToolProvider API.
     *
     * @return A new instance of a JavaCompiler.
     */
    static public JavaCompiler initCompiler() {
        final JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();

        if (compiler == null) {
            logger.error("Cannot find the system Java compiler. Check that Sun JDK is included in the classpath.");
            throw new RuntimeException("Cannot find the system Java compiler. Check that Sun JDK is included in the classpath.");
        }
        return compiler;
    }

    /**
     * Compiles the java file specified by the given {@link Path}.
     *
     * @param javaFile The {@link Path} of the java file.
     */
    public static void compileJavaFile(final JavaCompiler compiler, final String javaFile) {
        logger.info("Compiling Java file at {}.", javaFile);
        compiler.run(null, null, null, javaFile);
    }

    /**
     * Creates a new empty jar.
     *
     * @param directory The directory where to create the jar file.
     * @param filename  The name of the jar file without the extension.
     * @return The {@link JarOutputStream new empty jar file}.
     * @throws IOException If any problem occurs accessing the file system.
     */
    public static JarOutputStream createJar(final Path directory, final String filename) throws IOException {
        final String resultingName = directory.resolve(filename + "." + JAR_EXTENSION).toString();
        logger.info("Writing resulting jar file to {}.", resultingName);
        final FileOutputStream fout = new FileOutputStream(resultingName);
        return new JarOutputStream(fout);
    }

    /**
     * Writes the existing .class files on the directory into the jar file.
     *
     * @param exportDir The directory containing the .class files
     * @param jar       The jar to save the files to.
     * @return A boolean indicating whether the operation was successful or not.
     */
    public static boolean writeClassesToJar(final Path exportDir, final JarOutputStream jar) {
        final File[] files = exportDir.toFile().listFiles((dir, filename) -> filename.endsWith("." + JAVA_COMPILED_EXTENSION));

        Arrays.stream(Objects.requireNonNull(files))
                .forEach(classFile -> {
                    try {
                        jar.putNextEntry(new JarEntry(classFile.getName()));
                        com.google.common.io.Files.copy(classFile, jar);
                    } catch (final IOException e) {
                        final String msg = String.format("Problem writing file %s to jar.", classFile.getName());
                        logger.error(msg, e);
                        throw new RuntimeException(msg, e);
                    }
                });

        return true;
    }

    /**
     * Deletes unnecessary files from the given directory.
     *
     * @param exportDir The directory to clean.
     * @return A boolean indicating whether the operation was successful or not.
     */
    public static boolean cleanUnnecessaryFiles(final Path exportDir) {
        final File[] files = exportDir.toFile().listFiles((dir, filename) -> !filename.endsWith("." + JAR_EXTENSION));
        return Arrays.stream(Objects.requireNonNull(files)).allMatch(File::delete);
    }
}
