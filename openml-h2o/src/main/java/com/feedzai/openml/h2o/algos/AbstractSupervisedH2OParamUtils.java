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

import com.feedzai.openml.data.schema.DatasetSchema;
import water.api.schemas3.FrameV3;
import water.api.schemas3.KeyV3;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;

/**
 * Abstract class to parse H2O supervised algorithm params.
 *
 * @param <T> The concrete type of {@link ModelParametersSchemaV3 algorithm params}.
 * @since 1.0.0
 * @author Joao Sousa (joao.sousa@feedzai.com)
 */
public abstract class AbstractSupervisedH2OParamUtils<T extends ModelParametersSchemaV3> extends AbstractH2OParamUtils<T> {

    @Override
    protected T commonParams(final Frame trainingFrame, final DatasetSchema schema) {
        final T baseParams = getEmptyParams();
        baseParams.training_frame = new KeyV3.FrameKeyV3(trainingFrame._key);
        baseParams.ignore_const_cols = false;

        final int targetIndex = schema.getTargetIndex()
                .orElseThrow(() -> new IllegalArgumentException("Supervised algorithms require a target field."));

        final FrameV3.ColSpecifierV3 targetVar = new FrameV3.ColSpecifierV3();
        targetVar.column_name = trainingFrame.name(targetIndex);
        baseParams.response_column = targetVar;

        return baseParams;
    }

}
