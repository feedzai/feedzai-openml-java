/*
 * Copyright 2026 Feedzai
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
import org.junit.Test;

import java.net.URLClassLoader;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Tests for {@link JavaFileUtils}.
 */
public class JavaFileUtilsTest {

    /**
     * Validates that {@link JavaFileUtils#createNewInstanceFromClassLoader} throws
     * {@link ModelLoadingException} when the class cannot be found.
     */
    @Test
    public void testCreateNewInstanceFromClassLoaderThrowsOnInvalidClass() {
        final URLClassLoader classLoader = URLClassLoader.newInstance(
                new java.net.URL[0], getClass().getClassLoader()
        );

        assertThatThrownBy(() ->
                JavaFileUtils.createNewInstanceFromClassLoader(
                        "/fake/path/model.jar",
                        "com.nonexistent.%s.Model",
                        classLoader
                )
        ).isInstanceOf(ModelLoadingException.class)
         .hasMessageContaining("An error was found during the import of the model");
    }
}
