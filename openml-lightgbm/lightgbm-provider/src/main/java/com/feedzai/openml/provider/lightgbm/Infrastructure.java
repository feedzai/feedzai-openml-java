package com.feedzai.openml.provider.lightgbm;

import java.io.IOException;


/**
 * Enum that represents the infrastructure where code is running and consequent lgbm native libs locations.
 */
public class Infrastructure {

    /**
     * The CPU architecture used.
     */
    private final CpuArchitecture cpuArchitecture;

    /**
     * The libc implementation available.
     */
    private final LibcImplementation libcImpl;

    public Infrastructure(final CpuArchitecture cpuArchitecture, final LibcImplementation libcImpl) {
        this.cpuArchitecture = cpuArchitecture;
        this.libcImpl = libcImpl;
    }

    public String toString() {
        return cpuArchitecture.getLgbmNativeLibsFolder() + " architecture with " + libcImpl.getLibcImpl() + " libc implementation";
    }

    /**
     * Gets the native libraries folder name according to the cpu architecture and libc implementation.
     *
     * @return the native libraries folder name.
     */
    public String getLgbmNativeLibsFolder() throws IOException {

        if (libcImpl == LibcImplementation.MUSL && cpuArchitecture == CpuArchitecture.AARCH64) {
            throw new IOException("Trying to use LightGBM on a musl-based OS with unsupported arm64 architecture.");
        }

        final String libraryPath;
        if (cpuArchitecture == CpuArchitecture.AARCH64) {
            libraryPath = cpuArchitecture.getLgbmNativeLibsFolder() + "/";
        }
        else {
            libraryPath = cpuArchitecture.getLgbmNativeLibsFolder() + "/" + libcImpl.getLibcImpl() + "/";
        }

        return libraryPath;
    }
}
