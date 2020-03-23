# Feedzai OpenML Providers for Java
[![Build Status](https://travis-ci.com/feedzai/feedzai-openml-java.svg?branch=master)](https://travis-ci.com/feedzai/feedzai-openml-java)
[![codecov](https://codecov.io/gh/feedzai/feedzai-openml-java/branch/master/graph/badge.svg)](https://codecov.io/gh/feedzai/feedzai-openml-java)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/4d92197f37ab4811b81f34bd4847fee6?branch=master)](https://www.codacy.com/app/feedzai/feedzai-openml-java?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=feedzai/feedzai-openml-java&amp;utm_campaign=Badge_Grade)

Implementations of the Feedzai OpenML API to allow support for machine
learning models in Java. 


## Building
This is a Maven project which you can build using the following command:
```bash
mvn clean install
```

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
  <version>1.0.9</version>
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
  <version>1.0.9</version>
</dependency>
```
