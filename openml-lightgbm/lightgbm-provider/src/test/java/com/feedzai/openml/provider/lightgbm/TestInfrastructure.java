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

import java.io.IOException;

/**
 * Tests the retrieval of native libs folder path for Infrastructure.
 *
 * @author Renato Azevedo (renato.azevedo@feedzai.com)
 */
public class TestInfrastructure {
    @Test
    public void unknownInfrastructureCombination() {
        Infrastructure infrastructure = new Infrastructure(CpuArchitecture.AARCH64, LibcImplementation.MUSL);

        Assertions.assertThatThrownBy(infrastructure::getLgbmNativeLibsFolder)
                .isInstanceOf(IOException.class);
    }

    @Test
    public void knowsCorrectInfrastructureCombination() throws IOException {

        Infrastructure infra_arm64_glibc = new Infrastructure(CpuArchitecture.AARCH64, LibcImplementation.GLIBC);
        Assertions.assertThat(infra_arm64_glibc.getLgbmNativeLibsFolder())
                .isEqualTo("arm64/");

        Infrastructure infra_amd64_glibc = new Infrastructure(CpuArchitecture.AMD64, LibcImplementation.GLIBC);
        Assertions.assertThat(infra_amd64_glibc.getLgbmNativeLibsFolder())
                .isEqualTo("amd64/glibc/");

        Infrastructure infra_amd64_musl = new Infrastructure(CpuArchitecture.AMD64, LibcImplementation.MUSL);
        Assertions.assertThat(infra_amd64_musl.getLgbmNativeLibsFolder())
                .isEqualTo("amd64/musl/");
    }
}
