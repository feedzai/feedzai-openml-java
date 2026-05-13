/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * © 2026 Feedzai, Strictly Confidential
 */

package com.feedzai.openml.provider.lightgbm;

import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.feedzai.openml.data.schema.DatasetSchema;

import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.SAMPLE_WEIGHT_COL_PARAMETER_NAME;

/**
 * Utility class to parse and resolve the sample weight column parameter for LightGBM model training.
 *
 * @author Joaquim Leitão (joaquim.leitao@feedzai.com)
 */
public class SampleWeightParamParserUtil {

    /**
     * This class is not meant to be instantiated.
     */
    private SampleWeightParamParserUtil() {
    }

    /**
     * Retrieves the name of the sample weight field specified in the model parameters.
     *
     * @param params LightGBM train parameters.
     * @return The name of the sample weight field, if specified in the model parameters;
     *         otherwise empty Optional
     */
    public static Optional<String> getSampleWeightFieldName(final Map<String, String> params) {
        final String softLabelFieldName = params.get(SAMPLE_WEIGHT_COL_PARAMETER_NAME);

        return softLabelFieldName == null || softLabelFieldName.isEmpty() ?
                Optional.empty() : Optional.of(softLabelFieldName.trim());
    }

    /**
     * Gets the (canonical) index of the constraint group column.
     * NOTE: the sample weight column must be part of the features in the Dataset, but it may be ignored for training
     *
     * @param params LightGBM train parameters.
     * @param schema Schema of the dataset.
     * @return the index of the sample weight column if one was provided, else returns an empty Optional.
     */
    public static Optional<Integer> getSampleWeightColumnIndex(final Map<String, String> params,
                                                               final DatasetSchema schema) {
        final Optional<String> sampleWeightFieldName = getSampleWeightFieldName(params);
        return sampleWeightFieldName.isPresent() ?
                SchemaFieldsUtil.getColumnIndex(sampleWeightFieldName.get(), schema) : Optional.empty();
    }
}
