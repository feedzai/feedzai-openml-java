/*
 * Copyright 2020 Feedzai
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

package com.feedzai.openml.provider.lightgbm;

import com.microsoft.ml.lightgbm.lightgbmlibJNI;

/**
 * Many calls to the SWIG LightGBM interface can error out.
 * This is a RuntimeException that by default fetches the
 * last error in that LightGBM SWIG backend.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 */
public class LightGBMException extends RuntimeException {

    /**
     * Creates a new LightGBM RuntimeException to be used
     * after a LightGBM SWIG call fails (returnCodeLGBM==-1).
     *
     * This will fetch the respective LightGBM error message
     * from the SWIG wrappers and return a new RuntimeException
     * with that message.
     *
     * Note: Requires lightgbmlibJNI to be loaded beforehand!
     * @see LightGBMUtils#loadLibs()
     */
    public LightGBMException() {
        super("LightGBM error: " + lightgbmlibJNI.LGBM_GetLastError());
    }
}
