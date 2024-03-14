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

    @Override
    public String toString() {
        return "Infrastructure{" +
                "cpuArchitecture=" + cpuArchitecture +
                ", libcImpl=" + libcImpl +
                '}';
    }

    /**
     * Gets the native libraries folder name according to the cpu architecture and libc implementation.
     *
     * @return the native libraries folder name.
     */
    public String getLgbmNativeLibsFolder() throws IOException {

        switch (cpuArchitecture) {
            case AARCH64:
                if (libcImpl == LibcImplementation.MUSL) {
                    throw new IOException("Trying to use LightGBM on a musl-based OS with unsupported arm64 architecture.");
                }
                return cpuArchitecture.getLgbmNativeLibsFolder() + "/";
            case AMD64:
                return cpuArchitecture.getLgbmNativeLibsFolder() + "/" + libcImpl.getLibcImpl() + "/";
            default:
                throw new IllegalStateException("Unexpected value: " + cpuArchitecture);
        }
    }
}
