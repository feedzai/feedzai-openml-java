package com.feedzai.openml.provider.lightgbm;

/**
 * Enum that represents the libc implementation available on the machine and consequent lgbm native libs locations.
 */
public enum LibcImplementation {
    MUSL("musl"),
    GLIBC("glibc");

    /**
     * This is the name of available libc implementation and indicates the folder where the lightgbm native libraries are.
     */
    private final String libcImpl;

    LibcImplementation(final String libcImpl){
        this.libcImpl = libcImpl;
    }

    /**
     * Gets the native libraries folder name according to the libc implementation.
     *
     * @return the native libraries folder name according to the libc implementation.
     */
    public String getLibcImpl() {
        return libcImpl;
    }
}
