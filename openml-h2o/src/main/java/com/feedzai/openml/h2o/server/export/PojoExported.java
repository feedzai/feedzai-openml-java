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
import com.feedzai.openml.java.utils.JavaFileUtils;
import hex.Model;

import javax.tools.JavaCompiler;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.jar.JarOutputStream;

/**
 * Concrete implementation of an {@link ExportedModel} that represents an H2O POJO.
 *
 * @author Pedro Rijo (pedro.rijo@feedzai.com)
 * @since 0.1.0
 */
public class PojoExported implements ExportedModel {

    /**
     * Extension of jar files.
     */
    public static final String POJO_EXTENSION = JavaFileUtils.JAR_EXTENSION;

    /**
     * An instance of a {@link JavaCompiler}.
     */
    private final JavaCompiler compiler;

    /**
     * Creates a new instance of this class.
     */
    public PojoExported() {
        this.compiler = JavaFileUtils.initCompiler();
    }

    @Override
    public void save(final Path exportDir, final Model model) throws IOException, ModelTrainingException {
        final String modelFilename = model._output._job._result.toString();

        final String javaFile = String.format(
                "%s%c%s%c%s",
                exportDir,
                File.separatorChar,
                modelFilename,
                '.',
                JavaFileUtils.JAVA_FILE_EXTENSION
        );
        try (final FileOutputStream fos = new FileOutputStream(javaFile);
             final BufferedOutputStream bos = new BufferedOutputStream(fos)) {

            model.toJava(bos, false, false);
        }

        /*
         * In theory, we should need to add the h2o-genmodel.jar to the classpath before compiling the POJO (.java) file.
         * But currently it is working without it.
         * We should also think if the file needs to be compiled  with a custom/specified class loader.
         * Useful link: https://stackoverflow.com/questions/1563909/how-to-set-classpath-when-i-use-javax-tools-javacompiler-compile-the-source
         * Also, pkernel MemoryCompiler may be useful on this subject.
         */
        JavaFileUtils.compileJavaFile(compiler, javaFile);
        try (final JarOutputStream jarOut = JavaFileUtils.createJar(exportDir, modelFilename)) {
            if (!JavaFileUtils.writeClassesToJar(exportDir, jarOut) || !JavaFileUtils.cleanUnnecessaryFiles(exportDir)) {
                throw new ModelTrainingException("Error compiling a Java file " + javaFile);
            }
        }
    }
}
