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

import com.microsoft.ml.lightgbm.SWIGTYPE_p_double;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_int;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.doubleChunkedArray;
import com.microsoft.ml.lightgbm.lightgbmlib;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;
import org.assertj.core.data.Offset;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for the ChunkedArray SWIG interface to ensure
 * it operates smoothly according to the API.
 *
 * @author Alberto Ferreira (alberto.ferreira@feedzai.com)
 * @since @@@next.release@@@
 */
public class SWIGChunkedArrayAPITest {

    /**
     * Must load the libraries so that the rest of the classes work.
     * Without the libraries instantiated, not even LightGBM exceptions can be thrown.
     */
    @BeforeClass
    static public void setupFixture() {
        /* Needed as we'll directly call the low-level ChunkedArray class,
         * without using the provider which usually handles this step for us.
         * To avoid depending on test execution order we load the libraries explicitly:         
         */
        LightGBMUtils.loadLibs();
    }

    /**
     * Test that using a ChunkedArray&ltdouble&gt works.
     * It is possible to:
     * <ul>
     *     <li>create</li>
     *     <li>add() values to it (generating multiple chunks)</li>
     *     <li>retrieve said values intact (assert through getitem())</li>
     * </ul>
     */
    @Test
    public void doubleChunkedArrayValuesOK() {
        final int max_i = 10;
        final long chunk_size = 3;
        final doubleChunkedArray chunkedArray = new doubleChunkedArray(chunk_size);

        for (int i = 1; i <= max_i; ++i) {
            chunkedArray.add(i * 1.1);
        }

        int chunk = 0;
        int pos = 0;
        for (int i = 0; i < max_i; ++i) {
            final double ref_value = (i+1) * 1.1;
            assertThat(chunkedArray.getitem(chunk, pos, -1))
                    .as("Value at chunk %d, position %d", chunk, pos)
                    .isCloseTo(ref_value, Offset.offset(1e-3));

            ++pos;
            if (pos == chunk_size) {
                ++chunk;
                pos = 0;
            }
        }
    }

    /**
     * Test that out of bounds accesses in any of the coordinates
     * results in the sentinel on_vail_value to be returned.
     */
    @Test
    public void doubleChunkedArrayOutOfBoundsError() {
        final double on_fail_sentinel_value = -1;
        final doubleChunkedArray chunkedArray = new doubleChunkedArray(3);

        // Test out of bounds chunk (only 1 exists, not 11):
        assertThat(chunkedArray.getitem(10, 0, on_fail_sentinel_value))
                .as("out-of-bounds return sentinel value")
                .isCloseTo(on_fail_sentinel_value, Offset.offset(1e-3));
        // Test out of bounds on first chunk:
        assertThat(chunkedArray.getitem(0, 10, on_fail_sentinel_value))
                .as("out-of-bounds return sentinel value")
                .isCloseTo(on_fail_sentinel_value, Offset.offset(1e-3));
    }

    /**
     * Assert that ChunkedArray's coalesce_to works.
     * Inserted elements should be returned the same and in the same
     * order at the output array.
     */
    @Test
    public void ChunkedArrayCoalesceTo() {
        final int numFeatures = 3;
        final int chunkSize = 2*numFeatures;  // Must be multiple
        final doubleChunkedArray chunkedArray = new doubleChunkedArray(chunkSize);
        // Fill 1 chunk + some part of other
        for (int i = 0; i < chunkSize + 1; ++i) {
            chunkedArray.add(i);
        }
        final SWIGTYPE_p_double swigArr = lightgbmlib.new_doubleArray(chunkedArray.get_add_count());

        chunkedArray.coalesce_to(swigArr);

        for (int i = 0; i < chunkedArray.get_add_count(); ++i) {
            double v = lightgbmlib.doubleArray_getitem(swigArr, i);
            assertThat(v).as("coalescedArray[%d]", i).isCloseTo(i, Offset.offset(1e-3));
        }
    }

    /**
     * Ensure that `LGBM_DatasetCreateFromMats` can be created
     * from the ChunkedArray.
     */
    @Test
    public void LGBM_DatasetCreateFromMatsFromChunkedArray() {
        final int numFeatures = 3;
        final int chunkSize = 2*numFeatures;  // Must be multiple
        final doubleChunkedArray chunkedArray = new doubleChunkedArray(chunkSize);
        // Fill 1 chunk + some part of other
        for (int i = 0; i < chunkSize + 1; ++i) {
            chunkedArray.add(i);
        }

        final long numChunks = chunkedArray.get_chunks_count();
        SWIGTYPE_p_int chunkSizes = lightgbmlib.new_intArray(numChunks);
        for (int i = 0; i < numChunks - 1; ++i) {
            lightgbmlib.intArray_setitem(chunkSizes, i, chunkSize);
        }
        lightgbmlib.intArray_setitem(chunkSizes, numChunks-1, (int)chunkedArray.get_current_chunk_added_count());

        final SWIGTYPE_p_p_void swigOutDatasetHandlePtr = lightgbmlib.voidpp_handle();;

        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetCreateFromMats(
                (int)chunkedArray.get_chunks_count(),
                chunkedArray.data_as_void(),
                lightgbmlibConstants.C_API_DTYPE_FLOAT64,
                chunkSizes,
                numFeatures,
                1,
                "", // parameters
                null,
                swigOutDatasetHandlePtr
        );

        assertThat(returnCodeLGBM).as("LightGBM return code").isEqualTo(0);
    }

}
