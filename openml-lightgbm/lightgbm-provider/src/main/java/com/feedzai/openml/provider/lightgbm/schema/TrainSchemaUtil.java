package com.feedzai.openml.provider.lightgbm.schema;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import com.feedzai.openml.provider.lightgbm.parameters.SoftLabelParamParserUtil;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static java.util.stream.Collectors.toList;

/**
 * Utils to account for features
 *
 * @author alberto.ferreira
 */
public class TrainSchemaUtil {

    private TrainSchemaUtil() {

    }

    /**
     * Gets the actual number of features to use for train.
     * If the soft label is used, it must be passed in the predictiveFields (features screen), but then excluded
     * from the features, or there would be label leakage, as it is the label during fit.
     *
     * @param schema train dataset Schema
     * @param params train parameters
     * @return the actual number of features (excluding soft label in case it is used to train - in which case it's removed from the features)
     */
    static public int getNumActualFeatures(final DatasetSchema schema, final Map<String, String> params) {
        final int rawNumPredictiveFields = schema.getPredictiveFields().size();

        final Optional<String> softLabelFieldName = SoftLabelParamParserUtil.getSoftLabelFieldName(params);

        if (!softLabelFieldName.isPresent()) {
            return rawNumPredictiveFields;
        }

        final boolean isSoftLabelSelectedAsFeature = schema
                .getPredictiveFields()
                .stream()
                .anyMatch(
                        field -> field.getFieldName()
                                .equals(softLabelFieldName.get()));

        return rawNumPredictiveFields - (isSoftLabelSelectedAsFeature ? 1 : 0);
    }

    static public String[] getActualFeatureNames(final DatasetSchema schema,
                                                     final Optional<String> softLabelFieldName) {

        if (!softLabelFieldName.isPresent()) {
            return getFieldNamesArray(schema.getPredictiveFields());
        }

        return getFieldNamesArray(
                schema.getPredictiveFields()
                        .stream()
                        .filter(field -> !field.getFieldName().equals(softLabelFieldName.orElse("")))
                        .collect(toList())
        );
    }


    /**
     * @param fields List of FieldSchema fields.
     * @return Names of the fields in the input list.
     */
    private static String[] getFieldNamesArray(final List<FieldSchema> fields) {
        return fields.stream().map(FieldSchema::getFieldName).toArray(String[]::new);
    }

    /**
     * @param fields List of FieldSchema fields.
     * @return Names of the fields in the input list.
     */
    private static List<String> getFieldNames(final List<FieldSchema> fields) {
        return Arrays.asList(getFieldNamesArray(fields));
    }
}
