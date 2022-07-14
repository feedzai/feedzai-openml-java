/*
 * Copyright 2022 Feedzai
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

import com.google.common.collect.ImmutableSet;
import java.util.EnumSet;
import java.util.Set;

import java.util.stream.Collectors;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * This class guarantees that no fairness-blind algorithm is used with FairGBMMLProvider.
 *
 * Conversely, it guarantees that no fairness-aware algorithm is used with LightGBMMLProvider.
 *
 * @author Sérgio Jesus (sergio.jesus@feedzai.com)
 * @since 1.4.0
 */
public class GBMProvidersTest {

    /**
     * Enum of fairness-aware algorithms.
     */
    private final Set<LightGBMAlgorithms> FAIRNESS_AWARE_ALGORITHMS = ImmutableSet.of(LightGBMAlgorithms.FAIRGBM_BINARY_CLASSIFIER);

    /**
     * Ensures models in FairGBMProvider are in the list of fair algorithms.
     *
     * @throws AssertionError In case there are algorithms which are not fair in provider.
     */
    @Test
    public void fairGBMProviderOnlyProvidesFairAlgorithms() {
        final FairGBMMLProvider fairGBMMLProvider = new FairGBMMLProvider();

        assertThat(
            fairGBMMLProvider.getAlgorithms().stream().filter(
                el -> FAIRNESS_AWARE_ALGORITHMS.contains(el)
            ).collect(Collectors.toSet()).size()).as("number of invalid algorithms").isEqualTo(0);
    }

    /**
     * Ensures models in LightGBMProvider are not in the list of fair algorithms.
     *
     * @throws AssertionError In case there are algorithms which are fair in provider.
     */
    @Test
    public void lightGBMProviderOnlyProvidesFairnessBlindAlgorithms(){
        final LightGBMMLProvider lightGBMMLProvider = new LightGBMMLProvider();

        final Set<LightGBMAlgorithms> fairnessBlindAlgorithms = EnumSet.allOf(LightGBMAlgorithms.class);
        fairnessBlindAlgorithms.removeAll(FAIRNESS_AWARE_ALGORITHMS);

        assertThat(
            lightGBMMLProvider.getAlgorithms().stream().filter(
                el -> fairnessBlindAlgorithms.contains(el)
            ).collect(Collectors.toSet()).size()).as("number of invalid algorithms").isEqualTo(0);
    }
}
