# Feedzai OpenML Providers for Java
[![Build Status](https://travis-ci.com/feedzai/feedzai-openml-java.svg?branch=hf-1.2.X)](https://travis-ci.com/feedzai/feedzai-openml-java)
[![codecov](https://codecov.io/gh/feedzai/feedzai-openml-java/branch/hf-1.2.X/graph/badge.svg)](https://codecov.io/gh/feedzai/feedzai-openml-java)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/4d92197f37ab4811b81f34bd4847fee6?branch=hf-1.2.X)](https://www.codacy.com/app/feedzai/feedzai-openml-java?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=feedzai/feedzai-openml-java&amp;utm_campaign=Badge_Grade)

Implementations of the Feedzai OpenML API to allow support for machine
learning models in Java. 

## Building
This is a Maven project which you can build using the following command:
```bash
mvn clean install
```

## Releasing

For all releases, as the hotfix branch is ready all that's needed to actually release is to create an annotated tag pointing to the hotfix branch head (example below for releasing version 1.2.29):

```bash
# Ensure the tag is made on the udpated branch
git fetch -a
git checkout origin/hf-1.2.X
git tag -a 1.2.29
# Your EDITOR will open. Write a good message and save as it is used on Github as a release message
git push origin 1.2.29
```
Then you need to [create a new release](https://github.com/feedzai/feedzai-openml-java/releases/new) with this tag and the description according [to the previous ones](https://github.com/feedzai/feedzai-openml-java/releases).

## Modules

### H2O
[![Maven metadata URI](https://img.shields.io/maven-metadata/v/http/central.maven.org/maven2/com/feedzai/openml-h2o/maven-metadata.xml.svg)](https://mvnrepository.com/artifact/com.feedzai/openml-h2o)

The `openml-h2o` module contains a provider that allows to load and train models with [H2O](https://www.h2o.ai/).

Pull the provider from Maven Central:
```xml
<dependency>
  <groupId>com.feedzai</groupId>
  <artifactId>openml-h2o</artifactId>
  <!-- See project tags for latest version -->
  <version>1.2.0</version>
</dependency>
```

### DataRobot
[![Maven metadata URI](https://img.shields.io/maven-metadata/v/http/central.maven.org/maven2/com/feedzai/openml-datarobot/maven-metadata.xml.svg)](https://mvnrepository.com/artifact/com.feedzai/openml-datarobot)

The `openml-datarobot` module contains a provider that allows to load models trained with [DataRobot](https://www.datarobot.com/).

Pull this module from Maven Central:
```xml
<dependency>
  <groupId>com.feedzai</groupId>
  <artifactId>openml-datarobot</artifactId>
  <!-- See project tags for latest version -->
  <version>1.2.0</version>
</dependency>
```

### LightGBM
[![Maven metadata URI](https://img.shields.io/maven-metadata/v/http/central.maven.org/maven2/com/feedzai/openml-lightgbm/maven-metadata.xml.svg)](https://mvnrepository.com/artifact/com.feedzai/openml-lightgbm)

The `openml-lightgbm` module contains a provider that allows to load models trained with [Microsoft LightGBM](https://github.com/microsoft/LightGBM).

Pull this module from Maven Central:
```xml
<dependency>
  <groupId>com.feedzai</groupId>
  <artifactId>openml-lightgbm</artifactId>
  <!-- See project tags for latest version -->
  <version>1.2.0</version>
</dependency>
```
