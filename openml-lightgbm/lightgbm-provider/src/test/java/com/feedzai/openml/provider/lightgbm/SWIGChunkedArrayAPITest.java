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
     * Test that using a ChunkedArray<double> works.
     * It is possible to:
     *  - create
     *  - add() values to it (generating multiple chunks)
     *  - retrieve said values intact (assert through getitem())
     */
    @Test
    public void doubleChunkedArrayValuesOK() {
        final int max_i = 10;
        final long chunk_size = 3;
        final doubleChunkedArray x = new doubleChunkedArray(chunk_size);

        for (int i = 1; i <= max_i; ++i) {
            x.add(i * 1.1);
        }

        int chunk = 0;
        int pos = 0;
        for (int i = 0; i < max_i; ++i) {
            final double ref_value = (i+1) * 1.1;
            assertThat(x.getitem(chunk, pos, -1))
                    .as("value")
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
        final doubleChunkedArray x = new doubleChunkedArray(3);

        // Test out of bounds chunk (only 1 exists, not 11):
        assertThat(x.getitem(10, 0, on_fail_sentinel_value))
                .as("output_sentinel")
                .isCloseTo(on_fail_sentinel_value, Offset.offset(1e-3));
        // Test out of bounds on first chunk:
        assertThat(x.getitem(0, 10, on_fail_sentinel_value))
                .as("output_sentinel")
                .isCloseTo(on_fail_sentinel_value, Offset.offset(1e-3));
    }

    @Test
    public void ChunkedArrayCoalesceTo() {
        final int numFeatures = 3;
        final int chunkSize = 2*numFeatures;  // Must be multiple
        final doubleChunkedArray x = new doubleChunkedArray(chunkSize);
        // Fill 1 chunk + some part of other
        for (int i = 0; i < chunkSize + 1; ++i) {
            x.add(i);
        }
        final SWIGTYPE_p_double swigArr = lightgbmlib.new_doubleArray(x.get_add_count());

        x.coalesce_to(swigArr);

        for (int i = 0; i < x.get_add_count(); ++i) {
            double v = lightgbmlib.doubleArray_getitem(swigArr, i);
            assertThat(v).as("value").isCloseTo(i, Offset.offset(1e-3));
        }
    }


    @Test
    public void LGBM_DatasetCreateFromMatsFromChunkedArray() {
        final int numFeatures = 3;
        final int chunkSize = 2*numFeatures;  // Must be multiple
        final doubleChunkedArray x = new doubleChunkedArray(chunkSize);
        // Fill 1 chunk + some part of other
        for (int i = 0; i < chunkSize + 1; ++i) {
            x.add(i);
        }

        final long numChunks = x.get_chunks_count();
        SWIGTYPE_p_int chunkSizes = lightgbmlib.new_intArray(numChunks);
        for (int i = 0; i < numChunks - 1; ++i) {
            lightgbmlib.intArray_setitem(chunkSizes, i, chunkSize);
        }
        lightgbmlib.intArray_setitem(chunkSizes, numChunks-1, (int)x.get_current_chunk_added_count());

        final SWIGTYPE_p_p_void swigOutDatasetHandlePtr = lightgbmlib.voidpp_handle();;

        final int returnCodeLGBM = lightgbmlib.LGBM_DatasetCreateFromMats(
                (int)x.get_chunks_count(),
                x.data_as_void(),
                lightgbmlibConstants.C_API_DTYPE_FLOAT64,
                chunkSizes,
                numFeatures,
                1,
                "", // parameters
                null,
                swigOutDatasetHandlePtr
        );

        assertThat(returnCodeLGBM).as("returnCode").isEqualTo(0);
    }

}
