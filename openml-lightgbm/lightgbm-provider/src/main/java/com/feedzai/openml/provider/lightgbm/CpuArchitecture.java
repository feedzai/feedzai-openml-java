/*
 * The copyright of this file belongs to Feedzai. The file cannot be
 * reproduced in whole or in part, stored in a retrieval system,
 * transmitted in any form, or by any means electronic, mechanical,
 * photocopying, or otherwise, without the prior permission of the owner.
 *
 * © 2023 Feedzai, Strictly Confidential
 */
package com.feedzai.openml.provider.lightgbm;

/**
 * Enum that represents the cpu architecture where code is running and consequent lgbm native libs locations.
 */
public enum CpuArchitecture {
    AARCH64("arm64"),
    AMD64("amd64");

    /**
     * This is the name of the folder where the lightgbm native libraries are.
     */
    private final String lgbmNativeLibsFolder;

    CpuArchitecture(final String lgbmNativeLibsFolder){
        this.lgbmNativeLibsFolder = lgbmNativeLibsFolder;
    }

    /**
     * Gets the native libraries folder name according to the cpu architecture.
     *
     * @return the native libraries folder name according to the cpu architecture.
     */
    public String getLgbmNativeLibsFolder() {
        return lgbmNativeLibsFolder;
    }
}
