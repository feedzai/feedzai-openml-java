package com.feedzai.openml.provider.lightgbm;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.feedzai.openml.data.schema.CategoricalValueSchema;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;

import static com.feedzai.openml.provider.lightgbm.FairGBMDescriptorUtil.CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME;

/**
 * Utility to parse parameters specific to the FairGBM model.
 *
 * @author Andre Cruz (andre.cruz@feedzai.com)
 * @since 1.3.6
 */
public class FairGBMParamParserUtil {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(FairGBMParamParserUtil.class);

    /**
     * Placeholder to use when an integer argument is not provided.
     * E.g., when running standard unconstrained LightGBM the constrained_group_column parameter will take this value.
     */
    static final int NO_SPECIFIC = -1;

    /**
     * String prefix used to refer to columns by name in LightGBM configs/parameters.
     */
    static final String COL_NAME_PREFIX = "name:";

    /**
     * This class is not meant to be instantiated.
     */
    private FairGBMParamParserUtil() {}

    /**
     * Whether the given mapParams correspond to a constrained LightGBM objective (aka FairGBM).
     * @param mapParams set of train parameters for LightGBM.
     * @return whether it is a fairness constrained objective.
     */
    public static boolean isFairnessConstrained(final Map<String, String> mapParams) {
        Optional<String> objective = LightGBMBinaryClassificationModelTrainer.getLightGBMObjective(mapParams);
        return (objective.isPresent() && objective.get().startsWith("constrained_"));
    }

    /**
     * Gets the (canonical) index of the constraint group column.
     * NOTE: the constraint group column must be part of the features in the Dataset, but it may be ignored for training
     * @param mapParams LightGBM train parameters.
     * @param schema Schema of the dataset.
     * @return the index of the constraint group column if one was provided, else returns an empty Optional.
     */
    public static Optional<Integer> getConstraintGroupColumnIndex(final Map<String, String> mapParams,
                                                                  final DatasetSchema schema) {
        // Parse the constraint_group_column, if one was provided
        String constraintGroupCol = mapParams.get(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME);
        if (constraintGroupCol == null || constraintGroupCol.trim().isEmpty()) {
            return Optional.empty();
        }

        // Trim white-space
        constraintGroupCol = constraintGroupCol.trim();

        // Remove the "name:" prefix
        final String actualConstraintGroupCol = constraintGroupCol.substring(
                constraintGroupCol.startsWith(COL_NAME_PREFIX) ? COL_NAME_PREFIX.length() : 0);

        // Find index for this column; consider label offset (LightGBM indices disregard the target column)
        final List<FieldSchema> featureFields = schema.getPredictiveFields();
        Optional<FieldSchema> constraintGroupField = featureFields
                .stream()
                .filter(field -> field.getFieldName().equalsIgnoreCase(actualConstraintGroupCol))
                .findFirst();

        // Check if column exists
        if (! constraintGroupField.isPresent()) {
            logger.error(String.format(
                    "The parameter %s=%s is invalid; no such column was found.",
                       CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME,
                       actualConstraintGroupCol));
            return Optional.empty();
        }

        // Check if the constraint_group_column is in categorical format
        if (! (constraintGroupField.get().getValueSchema() instanceof CategoricalValueSchema)) {
            logger.error(String.format(
                    "The parameter %s=%s is invalid; expected a column in categorical format, got %s format.",
                    CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME,
                    actualConstraintGroupCol,
                    constraintGroupField.get().getValueSchema().getClass().toString()));
            return Optional.empty();
        }

        // NOTE!
        //  - this index corresponds to the index in our dataset schema;
        //  - this value may be off by one when compared to LightGBM's expected index values;
        //  - this is due to the fact that LightGBM disregards the target column when counting column indices;
        final int constraintGroupColIndex = constraintGroupField.get().getFieldIndex();

        return Optional.of(constraintGroupColIndex);
    }

    /**
     * Gets the index of the constraint group column without the label column (LightGBM-specific format)
     * @param mapParams LightGBM train parameters.
     * @param schema Schema of the dataset.
     * @return the index of the constraint group column without the label if the constraint_group_column parameter
     * was provided, else returns an empty Optional.
     */
    public static Optional<Integer> getConstraintGroupColumnIndexWithoutLabel(final Map<String, String> mapParams,
                                                                              final DatasetSchema schema) {

        // Get canonical column index (including target column)
        final Optional<Integer> constraintGroupColumnIndex = getConstraintGroupColumnIndex(mapParams, schema);
        if (! constraintGroupColumnIndex.isPresent()) {
            return Optional.empty();
        }

        // Compute column index in LightGBM-specific format (disregarding target column)
        final int targetIndex = schema.getTargetIndex().get(); // Our model is supervised, and needs the target.
        final int fieldAfterLabelOffset = constraintGroupColumnIndex.get() > targetIndex ? -1 : 0;
        return Optional.of(constraintGroupColumnIndex.get() + fieldAfterLabelOffset);
    }
}
