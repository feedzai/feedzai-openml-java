# Microsoft LightGBM Provider

This is an implementation adapting Microsoft's LightGBM framework for Java focused in real-time environments.

The main goals of the implementation are:

 1. Reliability
 2. Low latency single-event scoring meant for realtime usage
 3. High throughput scoring
 4. High throughput train

## Installation

All you need is maven:

```bash
mvn clean install # or `mvn clean package` to generate the .jar alone
```

### Build time

The first build might take up to 15 minutes, depending on your connection speed. After that each build takes around 10-30 seconds.

The first build is lengthier as it needs to create [make-lightgbm](https://github.com/feedzai/make-lightgbm/)'s docker image and fetch maven dependencies.

## Changing LightGBM build version

To build against a different LightGBM version just edit `make.sh` and choose a different LightGBM tag/commit.

After that, delete the `lightgbmlib_build` folder prior to executing maven again.
