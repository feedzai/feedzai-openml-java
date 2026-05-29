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

package com.feedzai.openml.h2o.algos;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.feedzai.openml.h2o.H2OAlgorithm;
import com.feedzai.openml.h2o.algos.mocks.BindingFieldsNotArrayParameters;
import com.feedzai.openml.h2o.algos.mocks.BindingNoFieldsFieldParameters;
import com.feedzai.openml.h2o.algos.mocks.BindingPrivateFieldsFieldParameters;
import com.feedzai.openml.h2o.algos.mocks.BindingRegularParameters;
import com.feedzai.openml.h2o.algos.mocks.FieldsNotArrayParameters;
import com.feedzai.openml.h2o.algos.mocks.NoFieldsFieldParameters;
import com.feedzai.openml.h2o.algos.mocks.PrivateFieldsFieldParameters;
import com.feedzai.openml.h2o.algos.mocks.RegularParameters;
import com.feedzai.openml.h2o.params.ParametersBuilderUtil;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;

import org.junit.Test;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Java6Assertions.assertThat;


/**
 * Smoke test for H2O dynamic loading of model parameters.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 0.1.0
 */
public class ParametersTest {

    /**
     * Tests that all supported algorithms have at least some parameters as a kind of smoke test.
     */
    @Test
    public void testParametersExist() {

        Arrays.stream(H2OAlgorithm.values()).forEach(h2oAlg -> {
            final Set<ModelParameter> parameters = h2oAlg.getAlgorithmDescriptor().getParameters();

            assertThat(parameters)
                    .as("The parameters found")
                    .isNotEmpty();

            assertThat(parameters.size())
                    .as("The number of parameters")
                    .isGreaterThanOrEqualTo(5);

            assertThat(parameters.stream().anyMatch(ModelParameter::isMandatory))
                    .as("There is at least 1 mandatory parameter")
                    .isTrue();

            assertThat(parameters.stream().anyMatch(param -> !param.isMandatory()))
                    .as("There is at least 1 advanced parameter")
                    .isTrue();
        });
    }

    /**
     * Tests that a class with 3 fields declared but 2 in the fields array is filtered to return only the 2 fields in
     * the array.
     */
    @Test
    public void fieldNotInFieldsIsFiltered() {
        final Set<ModelParameter> parameters = ParametersBuilderUtil.getParametersFor(
                RegularParameters.class,
                BindingRegularParameters.class
        );

        assertThat(parameters)
                .as("The parameters found")
                .hasSize(2);

        final long parametersWithoutField3 = parameters.stream()
                .map(ModelParameter::getName)
                .filter(parameter -> !"field_3".equalsIgnoreCase(parameter))
                .count();

        assertThat(parametersWithoutField3)
                .as("Parameters filtered without field_3 are still 2")
                .isEqualTo(2);
    }

    /**
     * Tests that nothing is filtered if the {@link water.api.schemas3.ModelParametersSchemaV3} contains no static field
     * named {@code fields}.
     *
     * @since 1.0.7
     */
    @Test
    public void noFieldsField() {
        final Set<ModelParameter> parameters = ParametersBuilderUtil.getParametersFor(
                NoFieldsFieldParameters.class,
                BindingNoFieldsFieldParameters.class
        );

        assertThat(parameters)
                .as("The parameters found")
                .hasSize(2);
    }

    /**
     * Tests that an exception is thrown if the {@link water.api.schemas3.ModelParametersSchemaV3} has a private field.
     *
     * @since 1.0.7
     */
    @Test(expected = IllegalArgumentException.class)
    public void privateFieldsField() {
        ParametersBuilderUtil.getParametersFor(
                PrivateFieldsFieldParameters.class,
                BindingPrivateFieldsFieldParameters.class
        );
    }

    /**
     * Tests that a Warn is logged when a {@link water.api.schemas3.ModelParametersSchemaV3} has a {@code fields} field
     * that is not a String array.
     *
     * @since 1.0.7
     */
    @Test
    public void fieldsNotStringArray() {
        final ListAppender<ILoggingEvent> listAppender = appendLogger(ParametersBuilderUtil.class);

        ParametersBuilderUtil.getParametersFor(FieldsNotArrayParameters.class, BindingFieldsNotArrayParameters.class);

        final List<ILoggingEvent> loggingList = listAppender.list;

        final ILoggingEvent event = loggingList.get(0);

        assertThat(event.getLevel())
                .as("The logging level")
                .isEqualTo(Level.WARN);

        assertThat(event.getMessage())
                .as("The logged message")
                .containsIgnoringCase("is not a String Array as expected");
    }

    /**
     * Adds an appender to the given {@link Class} logger and returns it.
     *
     * @param clazz The class which logger will have an {@link ch.qos.logback.core.Appender} added.
     * @return The appender that was added to the given {@link Class} logger.
     * @since 1.0.7
     */
    private ListAppender<ILoggingEvent> appendLogger(final Class<?> clazz) {

        final Logger classLogger = (Logger) LoggerFactory.getLogger(clazz);

        final ListAppender<ILoggingEvent> listAppender = new ListAppender<>();
        listAppender.start();

        // add the appender to the logger
        // addAppender is outdated now
        classLogger.addAppender(listAppender);
        return listAppender;
    }

    /**
     * Tests that a case-mismatched enum name is still resolved.
     */
    @Test
    public void choiceFieldCaseMismatchCorrection() {
        final Set<ModelParameter> parameters = ParametersBuilderUtil.getParametersFor(
                MockCaseMismatchSchema.class,
                MockCaseMismatchBinding.class
        );

        assertThat(parameters).as("The choice parameter descriptor should be resolved successfully").hasSize(1);

        final ModelParameter parameter = parameters.iterator().next();

        assertThat(parameter.getName()).as("The parameter name should match the target field").isEqualTo("dummyField");

        final ChoiceFieldType fieldType = (ChoiceFieldType) parameter.getFieldType();

        assertThat(fieldType.getDefaultValue()).as("The enum 'VALUE_ONE' should match the schema's 'value_one' choice")
                                               .isEqualTo("value_one");
    }

    /**
     * Tests that when an enum field evaluates to null or does not match any valid choices, we fall back to the first
     * available choice.
     */
    @Test
    public void choiceFieldNullOrMissingFallback() {
        final ListAppender<ILoggingEvent> listAppender = appendLogger(ParametersBuilderUtil.class);

        final Set<ModelParameter> parameters = ParametersBuilderUtil.getParametersFor(
                MockFallbackSchema.class,
                MockFallbackBinding.class
        );

        assertThat(parameters).as("The parameter descriptor should be built using a fallback strategy").hasSize(1);

        final ModelParameter parameter = parameters.iterator().next();

        assertThat(parameter.getName()).as("The parameter name should match the target field").isEqualTo("dummyField");

        final ChoiceFieldType fieldType = (ChoiceFieldType) parameter.getFieldType();

        assertThat(fieldType.getDefaultValue()).as(
                "A null value should fall back safely to the first choice in the schema").isEqualTo("value_one");

        final List<ILoggingEvent> loggingList = listAppender.list;
        assertThat(loggingList).as("A fallback must log an explicit warning").isNotEmpty();

        final ILoggingEvent warningEvent = loggingList.get(0);
        assertThat(warningEvent.getLevel()).isEqualTo(Level.WARN);
        assertThat(warningEvent.getMessage()).as("The warning message should detail the fallback replacement")
                                             .containsIgnoringCase("is not present on possible values set");
    }

    /**
     * Dummy enum to simulate parameter choices.
     */
    public enum DummyEnum {
        VALUE_ONE, VALUE_TWO;
    }

    /**
     * Schema where the annotation choices are lowercase.
     */
    public static class MockCaseMismatchSchema extends water.api.schemas3.ModelParametersSchemaV3 {
        @water.api.API(help = "Case mismatch", values = {"value_one", "value_two"})
        public DummyEnum dummyField;
    }

    /**
     * Fallback schema to be used when client binding POJO is null.
     */
    public static class MockFallbackSchema extends water.api.schemas3.ModelParametersSchemaV3 {
        @water.api.API(help = "Fallback", values = {"value_one"})
        public DummyEnum dummyField;
    }

    /**
     * Client binding POJO where the runtime field resolves to uppercase.
     */
    public static class MockCaseMismatchBinding extends water.bindings.pojos.ModelParametersSchemaV3 {
        public DummyEnum dummyField = DummyEnum.VALUE_ONE;
    }

    /**
     * Client binding POJO where the enum initializes as null.
     */
    public static class MockFallbackBinding extends water.bindings.pojos.ModelParametersSchemaV3 {
        public DummyEnum dummyField = null;
    }
}
