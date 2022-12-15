package com.feedzai.openml.provider.lightgbm.parameters;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.data.schema.FieldSchema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.SOFT_LABEL_PARAMETER_NAME;

/**
 * Utility to parse parameters specific to the FairGBM model.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 */
public class SoftLabelParamParserUtil {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(SoftLabelParamParserUtil.class);

    /**
     * This class is not meant to be instantiated.
     */
    private SoftLabelParamParserUtil() {
    }

    public static Optional<String> getSoftLabelFieldName(final Map<String, String> params) {
        final String softLabelFieldName = params.get(SOFT_LABEL_PARAMETER_NAME);

        return softLabelFieldName.equals("") ? Optional.empty() : Optional.of(softLabelFieldName.trim());
    }

    public static boolean useSoftLabel(final Map<String, String> mapParams) {
        return getSoftLabelFieldName(mapParams).isPresent();
    }


    /**
     * Gets the (canonical) index of the constraint group column.
     * NOTE: the soft label column must be part of the features in the Dataset, but it may be ignored for training
     *
     * @param params LightGBM train parameters.
     * @param schema    Schema of the dataset.
     * @return the index of the soft label column if one was provided, else returns an empty Optional.
     */
    public static Optional<Integer> getSoftLabelColumnIndex(final Map<String, String> params,
                                                            final DatasetSchema schema) {
        final Optional<String> softLabelFieldName = getSoftLabelFieldName(params);
        return softLabelFieldName.isPresent() ?
                SchemaFieldsUtil.getColumnIndex(softLabelFieldName.get(), schema) : Optional.empty();
    }

    public static Optional<Integer> getSoftLabelFieldIndexWithoutLabel(final Map<String, String> params,
                                                                       final DatasetSchema schema) {
        final Optional<String> softLabelFieldName = getSoftLabelFieldName(params);
        return softLabelFieldName.isPresent() ?
                SchemaFieldsUtil.getFieldIndexWithoutLabel(softLabelFieldName.get(), schema) : Optional.empty();
    }
}
