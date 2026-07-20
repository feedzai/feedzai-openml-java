# Java 25 Migration — Changes Made

## Files Modified

### `pom.xml` (root)
- `project.source`: `1.8` → `11`
- Added: `<maven.compiler.release>11</maven.compiler.release>`
- `jmockit.version`: `1.35` → `1.49`
- `maven-compiler-plugin`: `3.7.0` → `3.13.0`
  - Replaced `<source>/<target>` with `<release>${maven.compiler.release}</release>`
  - Added `<proc>full</proc>` (required for JDK 23+)
- `jacoco-maven-plugin`: `0.8.4` → `0.8.12`
- `maven-surefire-plugin`: `3.0.0-M5` → `3.5.2`
  - Added `<argLine>` with `-javaagent` for JMockit (resolved via `${settings.localRepository}`) and `--add-opens` for module access
- `auto-service`: `1.0-rc2` → `1.1.1`

### `.mvn/extensions.xml`
- `jgitver-maven-plugin`: `1.5.1` → `1.9.0`

### `openml-java-utils/src/main/java/.../JavaFileUtils.java`
- Line 111: `urlClassLoader.loadClass(...).newInstance()` → `urlClassLoader.loadClass(...).getDeclaredConstructor().newInstance()`

### `openml-java-utils/src/test/java/.../ModelParameterUtilsTest.java`
- Removed `import mockit.integration.junit4.JMockit` and `import org.junit.runner.RunWith`
- Removed `@RunWith(JMockit.class)` annotation (JMockit 1.49 removed the runner class; mocking works via agent only)

### `openml-h2o/src/main/java/.../ParametersBuilderUtil.java`
- `getParamsInstance()`: `paramsClass.newInstance()` → `paramsClass.getDeclaredConstructor().newInstance()`
- Exception handling: `InstantiationException | IllegalAccessException` → broad `Exception` (covers `NoSuchMethodException`, `InvocationTargetException`)

### `.github/workflows/build.yml`
- Main build JDK: `8` (Zulu) → `17` (Temurin) — H2O 3.46 hard-rejects JDK 18+ at runtime
- musl Docker test: `openjdk8` → `openjdk17`
- arm64 Docker test: `maven:3.8-openjdk-8-slim` → `maven:3.9-eclipse-temurin-17`

## Version Bump Summary Table

| Component | From | To | Reason |
|---|---|---|---|
| maven-compiler-plugin | 3.7.0 | 3.13.0 | `--release` flag, JDK 23+ annotation processing |
| JaCoCo | 0.8.4 | 0.8.12 | JDK 25 bytecode instrumentation |
| Surefire | 3.0.0-M5 | 3.5.2 | JDK 25 fork support, argLine handling |
| Auto-Service | 1.0-rc2 | 1.1.1 | JDK 23+ annotation processing |
| JMockit | 1.35 | 1.49 | JDK 17+ agent attachment |
| jgitver | 1.5.1 | 1.9.0 | JDK 21+ runtime compatibility |
| Compiler target | 1.8 | 11 (release) | Minimum for JDK 25 toolchain |

## Runtime Requirements (Deploying on JDK 25)

When running on JDK 25, add these JVM flags:

```
--add-opens java.base/java.lang=ALL-UNNAMED
--add-opens java.base/java.lang.reflect=ALL-UNNAMED
--add-opens java.base/java.lang.invoke=ALL-UNNAMED
--add-opens java.base/java.util=ALL-UNNAMED
--add-opens java.base/java.io=ALL-UNNAMED
--add-opens java.base/java.net=ALL-UNNAMED
--add-opens java.base/java.nio=ALL-UNNAMED
--add-opens java.base/java.math=ALL-UNNAMED
--add-opens java.base/sun.nio.ch=ALL-UNNAMED
--add-opens java.base/java.util.concurrent=ALL-UNNAMED
--add-opens java.base/java.util.concurrent.locks=ALL-UNNAMED
```

These are needed by:
- **H2O 3.46.0.11**: Deep reflection into JDK internals for serialization and memory management
- **DataRobot** (via commons-lang3 `FieldUtils`): `setAccessible(true)` to read `classLabels` field
- **JMockit 1.49**: Bytecode instrumentation for mocking
