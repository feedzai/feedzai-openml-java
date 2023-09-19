/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * © 2023 Feedzai, Strictly Confidential
 */
package com.feedzai.openml.provider.lightgbm;

import org.assertj.core.api.Assertions;
import org.junit.Test;

/**
 * Tests the retrieval of cpu architecture.
 *
 * @author Artur Pedroso (artur.pedroso@feedzai.com)
 */
public class TestCpuArchitecture {
    @Test
    public void unknownCpuArchitectureThrowsException() {
        Assertions.assertThatThrownBy(() -> LightGBMUtils.getCpuArchitecture("x86_64"))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    public void canReadKnownCpuArchitectures() {
        Assertions.assertThat(LightGBMUtils.getCpuArchitecture("aarch64"))
                .isEqualTo(CpuArchitecture.AARCH64);

        Assertions.assertThat(LightGBMUtils.getCpuArchitecture("amd64"))
                .isEqualTo(CpuArchitecture.AMD64);
    }
}
