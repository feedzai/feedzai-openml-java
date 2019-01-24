/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * © 2019 Feedzai, Strictly Confidential
 */

package com.feedzai.openml.h2o.algos;

import com.feedzai.openml.data.schema.DatasetSchema;
import water.api.schemas3.FrameV3;
import water.api.schemas3.KeyV3;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;

import java.util.Map;

/**
 * Abstract class to parse H2O supervised algorithm params.
 *
 * @param <T> The concrete type of {@link ModelParametersSchemaV3 algorithm params}.
 * @author Joao Sousa (joao.sousa@feedzai.com)
 * @since 1.0.0
 */
public abstract class AbstractUnsupervisedH2OParamUtils<T extends ModelParametersSchemaV3> extends AbstractH2OParamUtils<T> {

    /**
     * Auxiliary method to setup the common training params to all algorithms/models.
     *
     * @param trainingFrame The dataset to be used.
     * @param datasetSchema The schema correspondent to the training frame used.
     * @return A modified version of the provided params object.
     */
    private T commonParams(final Frame trainingFrame, final DatasetSchema datasetSchema) {
        final T baseParams = getEmptyParams();
        baseParams.training_frame = new KeyV3.FrameKeyV3(trainingFrame._key);
        baseParams.ignore_const_cols = false;

        datasetSchema.getTargetIndex().ifPresent(targetIndex -> {
            final FrameV3.ColSpecifierV3 targetVar = new FrameV3.ColSpecifierV3();
            targetVar.column_name = trainingFrame.name(targetIndex);
            baseParams.response_column = targetVar;
        });

        return baseParams;
    }

    /**
     * Template method to parse H2O algorithm params.
     *
     * @param trainingFrame The dataset to be used.
     * @param params        The raw training params.
     * @param randomSeed    The source of randomness.
     * @param datasetSchema
     * @return The modified version of the given {@code h2oParams}.
     */
    public final T parseParams(final Frame trainingFrame,
                               final Map<String, String> params,
                               final long randomSeed,
                               final DatasetSchema datasetSchema) {
        final T baseParams = commonParams(trainingFrame, datasetSchema);
        return parseSpecificParams(baseParams, params, randomSeed);
    }

}
