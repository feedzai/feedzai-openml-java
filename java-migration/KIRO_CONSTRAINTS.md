# Java 25 Migration — Constraints & Risks

## Hard Constraints

### 1. H2O Version Locked to 3.46.0.11
H2O version is determined by Pulse model compatibility. Cannot be bumped independently. H2O uses deep reflection (Unsafe, setAccessible) for its internal serialization — requires `--add-opens` at runtime.

### 2. DataRobot Prediction Library Uses Deep Reflection
`DataRobotModelCreator.getTargetModelValues()` uses `FieldUtils.readField(predictor, "classLabels", true)` which calls `setAccessible(true)`. This fails without `--add-opens java.base/java.lang.reflect=ALL-UNNAMED`. The DataRobot prediction JAR (1.1.1) is old and unlikely to get fixes.

### 3. LightGBM Is Pure JNI — No Java-Level Issues
LightGBM provider loads a native `.so` library via JNI. The Java wrapper code is straightforward and has no reflection or internal API usage. No issues expected on JDK 25.

### 4. Pulse Must Provide `--add-opens` at Runtime
The `--add-opens` flags MUST be configured in Pulse's `startup.sh` and `pulse.properties` (`pulse.global.appengine.jvmopts`). Without them, H2O model training and DataRobot model loading will fail with `InaccessibleObjectException`.

### 5. Alpine/musl: No JDK 25 Package Yet
Alpine Linux packages only up to OpenJDK 21 in stable repos. The musl CI test uses JDK 21 as a compromise. Full JDK 25 musl testing requires building from Temurin source or using a custom Docker image.

## Risks

### Medium: Jackson 2.6.7 Is Very Old
This project pins Jackson 2.6.7 (from 2016). While it compiles fine, it's deeply EOL. Jackson 2.6.x has known CVEs. However, it's provided-scope (Pulse supplies its own Jackson at runtime), so this is a classpath concern, not a vulnerability in this project's artifact.

### Medium: JMockit 1.49 Is the Last Version
JMockit is discontinued. It works on JDK 25 today via `--add-opens`, but future JDK versions may break it further. Long-term plan should migrate tests to Mockito.

### Low: Auto-Service Raw Type Warning
Auto-Service 1.1.1 emits a warning when `@AutoService(MachineLearningProvider.class)` is used on a generic implementation. This is cosmetic and can be suppressed with `@SuppressWarnings("rawtypes")` if desired.

### Low: Bytecode Target is 11, Not 25
Using `release=11` means the bytecode won't exercise JDK 25-specific runtime optimizations. This is intentional — the project produces library JARs loaded as plugins, and lower bytecode targets give broader compatibility.

## Compatibility Matrix

| JDK Version | Compile | Run (with --add-opens) | Run (without --add-opens) |
|---|---|---|---|
| 8 | ❌ (needs release 11 toolchain) | ❌ (bytecode target 11) | ❌ |
| 11 | ✅ | ✅ (mostly — H2O may warn) | ⚠️ (warnings only on JDK 11-15) |
| 17 | ✅ | ✅ | ❌ (InaccessibleObjectException) |
| 21 | ✅ | ✅ | ❌ |
| 25 | ✅ | ✅ | ❌ |
