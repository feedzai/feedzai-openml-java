#!/usr/bin/env bash

if [ "$TRAVIS_PULL_REQUEST" = "false" ] && [ ! -z "$TRAVIS_TAG" ]; then
    mvn deploy -Prelease --settings .m2/settings.xml -DskipTests=true
    echo "uploading make-lightgbm jar..."
    mvn deploy:deploy-file -Prelease --settings .m2/settings.xml -DskipTests=true -DgeneratePom=false -DpomFile=openml-lightgbm-meta-module/lightgbm-builder/make-lightgbm/build/pom.xml -Dfile=openml-lightgbm-meta-module/lightgbm-builder/make-lightgbm/build/lightgbmlib.jar
    exit $?
fi
