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
 * Tests the retrieval of libc implementation.
 *
 * @author Renato Azevedo (renato.azevedo@feedzai.com)
 */
public class TestLibcImplementation {
    @Test
    public void unknownLibcImplementationThrowsException() {
        Assertions.assertThatThrownBy(() -> LightGBMUtils.getLibcImplementation("klibc"))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    public void defaultLibcImplementationIsGlibc() {
        Assertions.assertThat(LightGBMUtils.getLibcImplementation(""))
                .isEqualTo(LibcImplementation.GLIBC);

        Assertions.assertThat(LightGBMUtils.getLibcImplementation(null))
                .isEqualTo(LibcImplementation.GLIBC);
    }

    @Test
    public void canReadKnownLibcImplementation() {
        Assertions.assertThat(LightGBMUtils.getLibcImplementation("glibc"))
                .isEqualTo(LibcImplementation.GLIBC);

        Assertions.assertThat(LightGBMUtils.getLibcImplementation("musl"))
                .isEqualTo(LibcImplementation.MUSL);
    }
}
