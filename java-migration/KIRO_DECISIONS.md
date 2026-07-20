# Java 25 Migration — Decisions

## Context
feedzai-openml-java provides ML model providers (H2O, LightGBM, DataRobot) for the Pulse platform. Pulse is migrating from Java 8 to Java 25, requiring all plugin JARs to compile and run on the new JDK.

## Key Decisions

### 1. Compiler Target: `release=11` (not 25)

**Decision**: Use `--release 11` instead of `--release 25`.

**Rationale**: The bytecode produced with `release=11` runs on any JDK ≥11, including JDK 25. This preserves backward compatibility for environments that aren't yet on JDK 25 while being fully forward-compatible. There is no API in this codebase that requires JDK 25 source features. The `--release` flag (introduced in JDK 9) prevents accidental use of APIs added after the target version.

**Trade-off**: Cannot use Java 12+ language features (records, sealed classes, pattern matching). Acceptable since this project is a stable library with minimal source changes.

### 2. JMockit 1.35 → 1.49

**Decision**: Bump JMockit to 1.49 (the last version).

**Rationale**: JMockit 1.35 does not attach as a Java agent correctly on JDK 17+. Version 1.49 works on JDK 25 with proper `--add-opens` and `-javaagent` configuration. Full removal to Mockito is deferred (too much test rewrite for this scope).

### 3. Auto-Service 1.0-rc2 → 1.1.1

**Decision**: Bump to 1.1.1 (latest stable).

**Rationale**: JDK 23+ disables implicit annotation processing by default (must use `-proc:full` or `annotationProcessorPaths`). Old auto-service 1.0-rc2 has compatibility issues with newer annotation processing APIs. 1.1.1 handles the new annotation processing model correctly.

### 4. `Class.newInstance()` → `getDeclaredConstructor().newInstance()`

**Decision**: Replace the deprecated pattern in two locations.

**Rationale**: `Class.newInstance()` has been deprecated since Java 9 and is marked for removal. The replacement properly propagates checked exceptions and is the canonical approach.

### 5. `--add-opens` for Test Execution

**Decision**: Add broad `--add-opens` to Surefire for test execution.

**Rationale**: JMockit, H2O internals, and DataRobot (via commons-lang3 FieldUtils) all use deep reflection. Without `--add-opens`, tests fail with `InaccessibleObjectException` on JDK 17+. The opens are: `java.lang`, `java.lang.reflect`, `java.lang.invoke`, `java.util`, `java.io`, `java.net`, `java.nio`, `java.math`, `sun.nio.ch`, `java.util.concurrent`, `java.util.concurrent.locks`.

### 6. JaCoCo 0.8.4 → 0.8.12

**Decision**: Bump JaCoCo.

**Rationale**: JaCoCo 0.8.4 cannot instrument bytecode produced by or running on JDK 17+. Version 0.8.12 supports up to JDK 25 class file format.

### 7. jgitver 1.5.1 → 1.9.0

**Decision**: Bump the Maven extension for version inference.

**Rationale**: jgitver 1.5.1 uses JGit APIs that have compatibility issues with newer JDKs. Version 1.9.0 is actively maintained and tested with JDK 21+.

### 8. maven-compiler-plugin 3.7.0 → 3.13.0

**Decision**: Bump the compiler plugin.

**Rationale**: The old 3.7.0 does not support the `--release` flag properly and has no awareness of JDK 23+ annotation processing changes. 3.13.0 adds `<proc>full</proc>` support and proper handling of newer JDK toolchains.

### 9. Surefire 3.0.0-M5 → 3.5.2

**Decision**: Bump Surefire.

**Rationale**: Surefire 3.0.0-M5 has known issues with `argLine` handling on newer JDKs. 3.5.2 is the latest stable with proper JDK 25 process forking support.

### 10. H2O 3.46.0.11 — NOT Bumped

**Decision**: Keep H2O at 3.46.0.11.

**Rationale**: H2O's version is dictated by Pulse's model compatibility requirements. The existing version compiles fine as a dependency — any reflection issues are runtime-only and handled by `--add-opens`.
