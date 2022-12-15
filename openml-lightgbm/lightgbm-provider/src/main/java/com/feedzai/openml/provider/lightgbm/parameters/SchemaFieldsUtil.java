package com.feedzai.openml.provider.lightgbm.parameters;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Optional;

/**
 * Utilities to compute fields/column indices.
 *
 * @author alberto.ferreira
 */
public class SchemaFieldsUtil {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(SchemaFieldsUtil.class);

    /**
     * Placeholder to use when an integer argument is not provided.
     * E.g., when running standard unconstrained LightGBM the constrained_group_column parameter will take this value.
     */
    public static final int NO_SPECIFIC = -1;

    public static Optional<Integer> getColumnIndex(final String fieldName,
                                                   final DatasetSchema schema) {

        final List<FieldSchema> featureFields = schema.getPredictiveFields();
        Optional<FieldSchema> field = featureFields
                .stream()
                .filter(field_ -> field_.getFieldName().equalsIgnoreCase(fieldName))
                .findFirst();

        // Check if column exists
        if (!field.isPresent()) {
            logger.error(String.format(
                    "Column %s was not found in the dataset.",
                    fieldName));
            return Optional.empty();
        }

        return Optional.of(field.get().getFieldIndex());
    }

    /**
     * Gets the index of the soft label column without the label column (LightGBM-specific format)
     *
     * @param fieldName Name of the field in the dataset.
     * @param schema    Schema of the dataset.
     * @return the index of the constraint group column without the label if the constraint_group_column parameter
     * was provided, else returns an empty Optional.
     */
    public static Optional<Integer> getFieldIndexWithoutLabel(final String fieldName,
                                                              final DatasetSchema schema) {

        final Optional<Integer> fieldIndex = getColumnIndex(fieldName, schema);
        if (!fieldIndex.isPresent()) {
            return Optional.empty();
        }

        // Compute column index in LightGBM-specific format (disregarding target column)
        final int targetIndex = schema.getTargetIndex()
                .orElseThrow(RuntimeException::new); // Our model is supervised. It needs the target.

        final int offsetIfFieldIsAfterLabel = fieldIndex.get() > targetIndex ? -1 : 0;
        return Optional.of(fieldIndex.get() + offsetIfFieldIsAfterLabel);
    }
}
