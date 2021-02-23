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

import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class is responsible for initializing, managing and releasing all
 * LightGBM SWIG train resources and resource handlers in a memory-safe manner.
 * <p>
 * Whatever happens, it guarantees that no memory leaks are left behind.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since 1.1.2 (renamed from SWIGTrainResources)
 */
class SWIGTrainBooster implements AutoCloseable {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(SWIGTrainBooster.class);

    /**
     * SWIG pointer to the output LightGBM Booster Handle during Booster structure instantiation.
     */
    public SWIGTYPE_p_p_void swigOutBoosterHandlePtr;

    /**
     * Handle of the LightGBM boosting model post-instantiation.
     */
    public SWIGTYPE_p_void swigBoosterHandle;

    /**
     * Constructor.
     * <p>
     * Allocates all the initial handles necessary to bootstrap (but not use) the
     * in-memory LightGBM dataset + booster structures.
     * <p>
     * After that the BoosterHandle and the DatasetHandle will still need to be initialized at the proper times:
     *
     * @see SWIGTrainBooster#initSwigBoosterHandle()
     */
    public SWIGTrainBooster() {

        this.swigOutBoosterHandlePtr = lightgbmlib.voidpp_handle();
    }

    /**
     * Setup swigBoosterHandle after its structure was created in-memory.
     */
    public void initSwigBoosterHandle() {
        this.swigBoosterHandle = lightgbmlib.voidpp_value(this.swigOutBoosterHandlePtr);
    }

    /**
     * Release any allocated resources.
     * This operation is idempotent and can be safely called at any time as many times as you wish.
     */
    @Override
    public void close() {

        if (this.swigOutBoosterHandlePtr != null) {
            lightgbmlib.delete_voidpp(this.swigOutBoosterHandlePtr);
            this.swigOutBoosterHandlePtr = null;
        }

        if (this.swigBoosterHandle != null) {
            lightgbmlib.LGBM_BoosterFree(this.swigBoosterHandle);
            this.swigBoosterHandle = null;
        }
    }
}
