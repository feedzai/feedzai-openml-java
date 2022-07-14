package com.feedzai.openml.provider.lightgbm;

import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

import static com.feedzai.openml.provider.lightgbm.FairGBMDescriptorUtil.CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests for the FairGBMParamParserUtil class.
 *
 * @author Andre Cruz (andre.cruz@feedzai.com)
 * @since 1.4.0
 */
public class FairGBMParamParserUtilTest {

    /**
     * Default FairGBM parameters.
     */
    private static final Map<String, String> FAIRGBM_DEFAULT_PARAMS = TestParameters.getDefaultFairGBMParameters();

    /**
     * Default LightGBM parameters.
     */
    private static final Map<String, String> LIGHTGBM_DEFAULT_PARAMS = TestParameters.getDefaultLightGBMParameters();

    @Test
    public void FairGBMModelHasFairnessEnabled() {
        assertThat(FairGBMParamParserUtil.isFairnessConstrained(FAIRGBM_DEFAULT_PARAMS))
                .as("FairGBM default parameters have fairness enabled")
                .isTrue();
    }

    @Test
    public void LightGBMModelHasFairnessDisabled() {
        assertThat(FairGBMParamParserUtil.isFairnessConstrained(LIGHTGBM_DEFAULT_PARAMS))
                .as("LightGBM has fairness disabled")
                .isFalse();
    }

    @Test
    public void getConstraintGroupColumnFailsWhenAbsentFromParams() {
        // FairGBM default params do not contain a constraint_group_col parameter
        assertThat(FairGBMParamParserUtil.getConstraintGroupColumnIndex(
                FAIRGBM_DEFAULT_PARAMS,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END))
                .as("Trying to extract constraint group column without passing it as a parameter")
                .isNotPresent();
    }

    @Test
    public void getConstraintGroupColumnSucceedsWhenPresentInParamsAndSchema() {
        final Map<String, String> paramsWithConstraintGroupCol = new HashMap<>(FAIRGBM_DEFAULT_PARAMS);
        paramsWithConstraintGroupCol.put(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME, "sensitive_group");

        assertThat(FairGBMParamParserUtil.getConstraintGroupColumnIndex(
                paramsWithConstraintGroupCol,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END))
                .as("Trying to extract constraint group column without passing it as a parameter")
                .isPresent();
    }

    @Test
    public void getConstraintGroupColumnFailsWhenAbsentFromSchema() {
        final Map<String, String> paramsWithConstraintGroupCol = new HashMap<>(FAIRGBM_DEFAULT_PARAMS);
        paramsWithConstraintGroupCol.put(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME, "sensitive_group");

        assertThat(FairGBMParamParserUtil.getConstraintGroupColumnIndex(
                paramsWithConstraintGroupCol,
                TestSchemas.NUMERICALS_SCHEMA_WITH_LABEL_AT_END))   // This schema does not contain the sensitive col
                .as("Trying to extract constraint group column without passing it as a parameter")
                .isNotPresent();
    }

    @Test
    public void getConstraintGroupColumnFailsWhenNotCategorical() {
        final Map<String, String> paramsWithConstraintGroupCol = new HashMap<>(FAIRGBM_DEFAULT_PARAMS);
        paramsWithConstraintGroupCol.put(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME, "amount");

        assertThat(FairGBMParamParserUtil.getConstraintGroupColumnIndex(
                paramsWithConstraintGroupCol,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END))
                .as("Trying to extract constraint group column without passing it as a parameter")
                .isNotPresent();
    }

    @Test
    public void getConstraintGroupColumnTakesLabelIntoAccount() {
        final Map<String, String> paramsWithConstraintGroupCol = new HashMap<>(FAIRGBM_DEFAULT_PARAMS);
        paramsWithConstraintGroupCol.put(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME, "sensitive_group");

        Integer colIndexLabelAtEnd = FairGBMParamParserUtil.getConstraintGroupColumnIndexWithoutLabel(
                paramsWithConstraintGroupCol,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END).get();

        Integer colIndexLabelAtStart = FairGBMParamParserUtil.getConstraintGroupColumnIndexWithoutLabel(
                paramsWithConstraintGroupCol,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_START).get();

        assertThat(colIndexLabelAtEnd)
                .as("Trying to extract constraint group column without passing it as a parameter")
                .isEqualTo(colIndexLabelAtStart);
    }

    @Test
    public void getConstraintGroupColumnFailsWithIntegerInput() {
        final Map<String, String> paramsWithConstraintGroupCol = new HashMap<>(FAIRGBM_DEFAULT_PARAMS);
        paramsWithConstraintGroupCol.put(CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME, "7");

        assertThat(FairGBMParamParserUtil.getConstraintGroupColumnIndex(
                paramsWithConstraintGroupCol,
                TestSchemas.CATEGORICALS_SCHEMA_LABEL_AT_END))
                .as("Trying to extract constraint group column without passing it as a parameter")
                .isNotPresent();
    }

}
