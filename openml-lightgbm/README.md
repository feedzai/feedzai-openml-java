# Microsoft LightGBM Provider

This implementation is a "wrapper" to Microsoft's LightGBM framework for Java focused in performance for real-time environments.

The main goals of the implementation are:

1.  Reliability
2.  Low latency single-event scoring meant for realtime usage
3.  High throughput scoring
4.  High throughput train

## Installation

This module depends on other parts of `feedzai-openml-java`. At the repo root run:

```bash
mvn clean install # or `mvn clean package` to generate the .jar alone
```

With the whole project built, you can now build this module "alone". Use the same command in this folder.


### Build time

The first build is lengthier (might take up to 15 minutes) as it needs to create [make-lightgbm](https://github.com/feedzai/make-lightgbm/)'s docker image and fetch maven dependencies. After that each build takes around 10-30 seconds.

