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

package com.feedzai.openml.h2o.params;

import com.feedzai.openml.h2o.algos.H2ODeepLearningUtils;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import com.feedzai.openml.provider.descriptor.fieldtype.BooleanFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.FreeTextFieldType;
import com.feedzai.openml.provider.descriptor.fieldtype.ModelParameterType;
import com.feedzai.openml.provider.descriptor.fieldtype.NumericFieldType;
import com.google.common.collect.ImmutableList;
import com.google.gson.annotations.SerializedName;
import water.api.API;
import water.api.schemas3.ModelParametersSchemaV3;
import water.api.schemas3.SchemaV3;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import static java.lang.reflect.Modifier.isStatic;

/**
 * Utility that builds the parameters' description for H2O machine learning algorithms by inspecting its API classes
 * via reflection.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 0.1.0
 */
public final class ParametersBuilderUtil {

    /**
     * Predicate that filters out fields that are not annotated with the API H2O annotation or that are not for input.
     */
    private static final Predicate<Field> ALLOW_ONLY_INPUT_PARAMS = field -> field.getAnnotation(API.class) != null &&
                    (field.getAnnotation(API.class).direction() == API.Direction.INPUT || field.getAnnotation(API.class).direction() == API.Direction.INOUT);

    /**
     * Predicate that filters out fields that are declared in abstract classes that are not specific to a machine
     * learning algorithm.
     */
    private static final Predicate<Field> ALLOW_ONLY_RELEVANT_PARAMS = field ->
            !field.getDeclaringClass().equals(ModelParametersSchemaV3.class) && !field.getDeclaringClass().equals(SchemaV3.class);

    /**
     * Predicate that accepts only primitives, Strings and enums, as we do not know how to handle other types.
     */
    private static final Predicate<Field> ALLOW_ONLY_PRIMITIVES_AND_ENUMS = field ->
            field.getType().isEnum() || field.getType().isPrimitive() ||
                    // Exception for Deep Learning hidden layers parameter that is important but is not handled properly
                    // (retrofit uses an array of integers but tries to encode it as strings...and so we have to
                    // consider it manually.
                    (field.getType().isArray() && H2ODeepLearningUtils.HIDDEN.equals(field.getName()));

    /**
     * Private constructor for utility class.
     */
    private ParametersBuilderUtil() { }

    /**
     * Gets the functions capable of assigning any parameter value to H2O's REST API POJO field that matches it.
     *
     * @param algorithmClass The class of the ML algorithm whose parameters we want to set.
     * @param <T> The concrete type of the class.
     * @return The function capable of setting parameter values for the given algorithm.
     */
    public static <T extends ModelParametersSchemaV3> ParamsValueSetter<T> getParamSetters(final Class<T> algorithmClass) {
        final Map<String, Field> paramName2Field = getParamNameToFieldNameMapping(algorithmClass)
                .entrySet()
                .stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        entry -> getFieldFromSchemaIn(algorithmClass, entry.getValue())
                ));

        return (paramsObj, paramName, paramValue) -> {
            final Field paramField = paramName2Field.get(paramName);
            if (paramField == null) {
                throw new RuntimeException("Unknown field " + paramName + " in algorithm " + algorithmClass);
            }

            try {
                handleParamSetter(algorithmClass, paramsObj, paramName, paramValue, paramField);

            } catch (final IllegalAccessException e) {
                throw new RuntimeException("Could not assign value " + paramValue + " to field " + paramName + " in algorithm " + algorithmClass, e);
            }
        };
    }

    /**
     * Gets the parameters descriptions for the given algorithm.
     *
     * @param algorithmClass The class with the meta-description of the parameters.
     * @param paramsClass    The corresponding class with fields whose values are default values for each parameter.
     * @return The descriptor for each parameter.
     */
    public static Set<ModelParameter> getParametersFor(final Class<? extends ModelParametersSchemaV3> algorithmClass,
                                                       final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> paramsClass) {

        final Map<String, String> paramName2FieldName = getParamNameToFieldNameMapping(paramsClass);

        final List<String> usefulFields = getUsefulFields(algorithmClass, paramName2FieldName.keySet());

        return Arrays.stream(algorithmClass.getFields())
                .filter(ALLOW_ONLY_INPUT_PARAMS)
                .filter(ALLOW_ONLY_RELEVANT_PARAMS)
                .filter(ALLOW_ONLY_PRIMITIVES_AND_ENUMS)
                .map(field -> getModelParameter(paramsClass, paramName2FieldName, field))
                // Excluded because the parameter is deprecated by H2O but is still in the code.
                .filter(modelParam -> !"r2_stopping".equals(modelParam.getName()))
                // Excluded because we never set a calibration frame.
                .filter(modelParam -> !"calibrate_model".equals(modelParam.getName()))
                // We already pass a seed always whenever possible.
                .filter(modelParam -> !"seed".equals(modelParam.getName()))
                // We only want the fields that are present in the list of fields of the algorithm class.
                .filter(modelParam -> usefulFields.contains(modelParam.getName()))
                .collect(Collectors.toSet());
    }

    /**
     * Returns the fields that are set in the given {@linkplain ModelParametersSchemaV3 algorithmClass} as useful for
     * the algorithm.
     *
     * @param algorithmClass The algorithm class to get the useful fields.
     * @param allParameters  All the parameters to be added if useful fields is not provided by the class.
     * @return The {@link List} with the useful fields for the given algorithm class.
     * @since 1.0.7
     */
    private static List<String> getUsefulFields(final Class<? extends ModelParametersSchemaV3> algorithmClass,
                                                final Set<String> allParameters) {

        final Optional<Field> fieldsField = Arrays.stream(algorithmClass.getDeclaredFields())
                .filter(field -> isStatic(field.getModifiers()))
                .filter(field -> "fields".equalsIgnoreCase(field.getName()))
                .findFirst();

        final ImmutableList.Builder<String> usefulFields = ImmutableList.builder();

        if (fieldsField.isPresent()) {
            usefulFields.add(getFieldContent(algorithmClass, fieldsField.get()));
        } else {
            usefulFields.addAll(allParameters);
        }

        return usefulFields.build();
    }

    /**
     * Returns an array with content of the given {@link Field} for the given {@linkplain ModelParametersSchemaV3
     * algorithmClass}.
     *
     * <p> If the given field is not a {@link String} array an empty array is returned.
     *
     * @param algorithmClass The class with the given {@code field}.
     * @param field          The field to retrieve the value.
     * @return The array with the field content.
     * @since 1.0.7
     */
    private static String[] getFieldContent(final Class<? extends ModelParametersSchemaV3> algorithmClass,
                                            final Field field) {

        final Class<?> fieldType = field.getType();

        if (fieldType.isArray() && fieldType.getComponentType().equals(String.class)) {
            try {
                return (String[]) field.get(null);
            } catch (final IllegalAccessException e) {
                throw new IllegalArgumentException(String.format(
                        "Unable to get useful fields for model of type: %s", algorithmClass.getTypeName()
                ));
            }
        }
        return new String[0];
    }

    /**
     * Gets the parameter for the given field.
     *
     * @param paramsClass         The class of the algorithm's parameters.
     * @param paramName2FieldName The mapping of parameter name to its field name.
     * @param field               The field for the parameter.
     * @return The model parameter descriptor.
     */
    private static ModelParameter getModelParameter(final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> paramsClass,
                                                    final Map<String, String> paramName2FieldName,
                                                    final Field field) {

        final Optional<String> optFieldName = Optional.ofNullable(paramName2FieldName.get(field.getName()));

        if (!optFieldName.isPresent()) {
            throw new RuntimeException("Could not match field " + field.getName() + " in fields of " + paramsClass);
        }

        final API apiAnnot = field.getAnnotation(API.class);
        final Class<?> fieldType = field.getType();
        final String fieldName = optFieldName.get();

        final ModelParameterType paramType;

        if (apiAnnot.values().length > 0) {
            final Set<String> possibleValues = Arrays.stream(apiAnnot.values()).collect(Collectors.toSet());
            final Enum<?> defaultValue = getDefaultChoiceValue(paramsClass, fieldName);
            paramType = new ChoiceFieldType(possibleValues, defaultValue.name());

        } else if (fieldType.equals(String.class)) {
            final String defaultValue = getDefaultStringValue(paramsClass, fieldName);
            paramType = new FreeTextFieldType(defaultValue);

        } else if (fieldType.equals(Boolean.class) || fieldType.equals(boolean.class)) {
            final boolean defaultValue = getDefaultBooleanValue(paramsClass, fieldName);
            paramType = new BooleanFieldType(defaultValue);

        } else if (fieldType.isArray() && H2ODeepLearningUtils.HIDDEN.equals(fieldName)) {
            paramType = new FreeTextFieldType("[200,200]");

        } else {

            final NumericFieldType.ParameterConfigType numberType;
            final double minVal;
            final double maxVal;

            if (fieldType.equals(Double.class) || fieldType.equals(Float.class) || fieldType.equals(double.class) || fieldType.equals(float.class)) {
                numberType = NumericFieldType.ParameterConfigType.DOUBLE;
                minVal = -Double.MAX_VALUE;
                maxVal = Double.MAX_VALUE;
            } else {
                numberType = NumericFieldType.ParameterConfigType.INT;
                minVal = -Integer.MAX_VALUE;
                maxVal = Integer.MAX_VALUE;
            }

            final double defaultValue = getDefaultNumericalValue(paramsClass, fieldName);
            paramType = NumericFieldType.range(minVal, maxVal, numberType, defaultValue);
        }

        return new ModelParameter(
                field.getName(),
                field.getName(),
                apiAnnot.help(),
                apiAnnot.level() == API.Level.critical,
                paramType
        );
    }

    /**
     * Gets a mapping from the names of the parameters to the corresponding names of the fields of the class that
     * represents the parameters in the H2O REST API.
     * <p>
     * I.e., the keys will be "parameter names" (as described in H2O API, e.g. "compute_metrics") and values will be
     * the corresponding field name in the POJO parameter (e.g., "computeMetrics").
     *
     * @apiNote This yields "false positives", in the sense that some entries of this map are not actually parameters
     * of the algorithm. Therefore users of the resulting map should query only for keys that are known to be
     * parameters.
     *
     * @param paramsClass The class whose fields contain the parameters for the algorithm.
     * @return The mapping.
     */
    private static Map<String, String> getParamNameToFieldNameMapping(final Class<?> paramsClass) {
        return Arrays.stream(paramsClass.getFields())
                    .collect(Collectors.toMap(
                            paramField -> paramField.getAnnotation(SerializedName.class) != null ?
                                    paramField.getAnnotation(SerializedName.class).value() : paramField.getName(),
                            Field::getName
                    ));
    }

    /**
     * Gets the default value for the given numerical field.
     *
     * @param defaultValuesClass    The class with the field.
     * @param defaultValueFieldName The name of the numerical field.
     * @return The numerical default value.
     */
    private static double getDefaultNumericalValue(final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> defaultValuesClass,
                                                   final String defaultValueFieldName) {

        final water.bindings.pojos.ModelParametersSchemaV3 defaultsInstance = getParamsInstance(defaultValuesClass);

        try {
            final Field field = getFieldFromPojoIn(defaultValuesClass, defaultValueFieldName);
            final Class<?> type = field.getType();
            if (type.equals(Integer.class) || type.equals(int.class)) {
                return field.getInt(defaultsInstance);

            } else if (type.equals(Short.class) || type.equals(short.class)) {
                return field.getShort(defaultsInstance);

            } else if (type.equals(Double.class) || type.equals(double.class)) {
                double doubleValue = field.getDouble(defaultsInstance);
                final double maximumValue = 17976931348600000000.0;
                if (doubleValue > maximumValue) {
                    // TODO PULSEDEV-23883 There is a current limitation in user interfaces accepting large values
                    // so we are avoiding these defaults temporarily to overcome that.
                    doubleValue = maximumValue;
                }
                return doubleValue;

            } else if (type.equals(Long.class) || type.equals(long.class)) {
                return field.getLong(defaultsInstance);

            } else if (type.equals(Float.class) || type.equals(float.class)) {
                return field.getFloat(defaultsInstance);

            } else {
                throw new RuntimeException("Unexpected type for numerical field " + field);
            }

        } catch (final IllegalAccessException e) {
            throw new RuntimeException("Could not get numerical value of field " + defaultValueFieldName + " in " + defaultValueFieldName, e);
        }
    }

    /**
     * Gets the default value for the given choice field.
     *
     * @param defaultValuesClass    The class with the field.
     * @param defaultValueFieldName The name of the choice field.
     * @return The choice default value.
     */
    private static Enum<?> getDefaultChoiceValue(final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> defaultValuesClass,
                                                 final String defaultValueFieldName) {

        try {
            return (Enum<?>) getFieldFromPojoIn(defaultValuesClass, defaultValueFieldName).get(getParamsInstance(defaultValuesClass));
        } catch (final IllegalAccessException e) {
            throw new RuntimeException("Could not get Enum value of field " + defaultValueFieldName + " in " + defaultValueFieldName, e);
        }
    }

    /**
     * Gets the default value for the given boolean field.
     *
     * @param defaultValuesClass    The class with the field.
     * @param defaultValueFieldName The name of the boolean field.
     * @return The boolean default value.
     */
    private static boolean getDefaultBooleanValue(final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> defaultValuesClass,
                                                  final String defaultValueFieldName) {

        try {
            return getFieldFromPojoIn(defaultValuesClass, defaultValueFieldName).getBoolean(getParamsInstance(defaultValuesClass));
        } catch (final IllegalAccessException e) {
            throw new RuntimeException("Could not get boolean value of field " + defaultValueFieldName + " in " + defaultValueFieldName, e);
        }
    }

    /**
     * Gets the default value for the given text field.
     *
     * @param defaultValuesClass    The class with the field.
     * @param defaultValueFieldName The name of the text field.
     * @return The String default value.
     */
    private static String getDefaultStringValue(final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> defaultValuesClass,
                                                final String defaultValueFieldName) {

        try {
            return (String) getFieldFromPojoIn(defaultValuesClass, defaultValueFieldName).get(getParamsInstance(defaultValuesClass));
        } catch (final IllegalAccessException e) {
            throw new RuntimeException("Could not get String value of field " + defaultValueFieldName + " in " + defaultValueFieldName, e);
        }
    }

    /**
     * Gets the field with the given name in the given class from the {@link water.bindings.pojos.ModelParametersSchemaV3 java bindings API POJO}.
     *
     * @param paramsClass The class.
     * @param fieldName   The field name.
     * @return The field.
     * @throws RuntimeException If the field cannot be found.
     * @since 0.1.0
     */
    private static Field getFieldFromPojoIn(final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> paramsClass,
                                            final String fieldName) {
        try {
            return paramsClass.getField(fieldName);
        } catch (final NoSuchFieldException e) {
            throw new RuntimeException("Field " + fieldName + " does not exist in " + fieldName, e);
        }
    }

    /**
     * Gets the field with the given name in the given class from the {@link water.api.schemas3.ModelParametersSchemaV3 API POJO}.
     *
     * @param paramsClass The class.
     * @param fieldName   The field name.
     * @return The field.
     * @throws RuntimeException If the field cannot be found.
     * @since 0.1.0
     */
    private static Field getFieldFromSchemaIn(final Class<? extends water.api.schemas3.ModelParametersSchemaV3> paramsClass,
                                              final String fieldName) {
        try {
            return paramsClass.getField(fieldName);
        } catch (final NoSuchFieldException e) {
            throw new RuntimeException("Field " + fieldName + " does not exist in " + fieldName, e);
        }
    }

    /**
     * Gets an instance of the given parameters' class.
     *
     * @param paramsClass The class containing fields that are parameters for the ML model. We assume it has an empty
     *                    constructor so we can instantiate it.
     * @return The instance for the given class; it will have default values in its fields.
     */
    private static water.bindings.pojos.ModelParametersSchemaV3 getParamsInstance(final Class<? extends water.bindings.pojos.ModelParametersSchemaV3> paramsClass) {
        try {
            return paramsClass.newInstance();
        } catch (final InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Sets the parameter value in the given parameters object.
     *
     * @param algorithmClass The class representing the ML algorithm.
     * @param paramsObj      The object that holds the parameters.
     * @param paramName      The name of the parameter to set.
     * @param paramValue     The value for the parameter in its string representation.
     * @param paramField     The field for the parameter in the parameters' object.
     * @param <T> The concrete type of the ML algorithm.
     * @throws IllegalAccessException If the parameter field cannot be accessed via reflection.
     */
    private static <T extends ModelParametersSchemaV3> void handleParamSetter(
            final Class<T> algorithmClass,
            final T paramsObj,
            final String paramName,
            final String paramValue,
            final Field paramField) throws IllegalAccessException {

        final Class<?> fieldType = paramField.getType();

        if (fieldType.equals(boolean.class) || fieldType.equals(Boolean.class)) {
            final boolean booleanVal = Boolean.parseBoolean(paramValue);
            if (fieldType.isPrimitive()) {
                paramField.setBoolean(paramsObj, booleanVal);
            } else {
                paramField.set(paramsObj, booleanVal);
            }

        } else if (fieldType.equals(Integer.class) || fieldType.equals(int.class)) {
            final Integer intVal = (int) Double.parseDouble(paramValue);
            if (fieldType.isPrimitive()) {
                paramField.setInt(paramsObj, intVal);
            } else {
                paramField.set(paramsObj, intVal);
            }

        } else if (fieldType.equals(Long.class) || fieldType.equals(long.class)) {
            final Long longVal = (long) Double.parseDouble(paramValue);
            if (fieldType.isPrimitive()) {
                paramField.setLong(paramsObj, longVal);
            } else {
                paramField.set(paramsObj, longVal);
            }

        } else if (fieldType.equals(Double.class) || fieldType.equals(double.class)) {
            final Double doubleVal = Double.parseDouble(paramValue);
            if (fieldType.isPrimitive()) {
                paramField.setDouble(paramsObj, doubleVal);
            } else {
                paramField.set(paramsObj, doubleVal);
            }

        } else if (fieldType.equals(Float.class) || fieldType.equals(float.class)) {
            final Float floatVal = Float.parseFloat(paramValue);
            if (fieldType.isPrimitive()) {
                paramField.setFloat(paramsObj, floatVal);
            } else {
                paramField.set(paramsObj, floatVal);
            }

        } else if (fieldType.equals(Short.class) || fieldType.equals(short.class)) {
            final Short shortVal = (short) Double.parseDouble(paramValue);
            if (fieldType.isPrimitive()) {
                paramField.setShort(paramsObj, shortVal);
            } else {
                paramField.set(paramsObj, shortVal);
            }

        } else if (fieldType.equals(String.class)) {
            paramField.set(paramsObj, paramValue);

        } else if (Enum.class.isAssignableFrom(fieldType)) {
            final Enum enumValue = Enum.valueOf((Class<Enum>) fieldType, paramValue);
            paramField.set(paramsObj, enumValue);

        } else if (fieldType.isArray()) {

            handleArrayParamSetter(algorithmClass, paramsObj, paramName, paramValue, paramField);

        } else {
            throw new RuntimeException("Unknown field type " + fieldType + " for field " + paramName + " in algorithm " + algorithmClass);
        }
    }

    /**
     * Handles setting the value of a parameter of a ML algorithm for the particular case of the parameter being
     * represented as an array in the ML Parameters' object.
     *
     * @param algorithmClass The class representing the ML algorithm.
     * @param paramsObj      The object that holds the parameters.
     * @param paramName      The name of the parameter to set.
     * @param paramValue     The value for the parameter in its string representation.
     * @param paramField     The field for the parameter in the parameters' object.
     * @param <T> The concrete type of the ML algorithm.
     * @throws IllegalAccessException If the parameter field cannot be accessed via reflection.
     */
    private static <T extends ModelParametersSchemaV3> void handleArrayParamSetter(
            final Class<T> algorithmClass,
            final T paramsObj,
            final String paramName,
            final String paramValue,
            final Field paramField) throws IllegalAccessException {

        final Class<?> fieldType = paramField.getType();

        final Class<?> arrayType = fieldType.getComponentType();
        if (arrayType.equals(boolean.class)) {
            final boolean booleanVal = Boolean.parseBoolean(paramValue);
            paramField.set(paramsObj, new boolean[] { booleanVal });

        } else if (arrayType.equals(int.class)) {
            final Integer intVal = Integer.parseInt(paramValue);
            paramField.set(paramsObj, new int[] { intVal });

        } else if (arrayType.equals(double.class)) {
            final Double doubleVal = Double.parseDouble(paramValue);
            paramField.set(paramsObj, new double[] { doubleVal });

        } else if (arrayType.equals(long.class)) {
            final Long longVal = Long.parseLong(paramValue);
            paramField.set(paramsObj, new long[]{longVal});

        } else if (arrayType.equals(float.class)) {
            final Float floatVal = Float.parseFloat(paramValue);
            paramField.set(paramsObj, new float[] { floatVal });

        } else if (arrayType.equals(short.class)) {
            final Short shortVal = Short.parseShort(paramValue);
            paramField.set(paramsObj, new float[] { shortVal });

        } else {
            throw new RuntimeException("Unexpected array type " + arrayType + " for parameter " + paramName + " in algorithm " + algorithmClass);
        }
    }

}
