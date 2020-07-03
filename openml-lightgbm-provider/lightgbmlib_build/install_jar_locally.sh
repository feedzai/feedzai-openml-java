#!/usr/bin/env

mvn install:install-file -Dfile=lightgbmlib.jar \
                         -DpomFile=pom.xml \
                         -DgeneratePom=false
