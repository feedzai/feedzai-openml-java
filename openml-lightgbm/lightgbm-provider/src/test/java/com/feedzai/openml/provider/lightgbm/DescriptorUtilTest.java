package com.feedzai.openml.provider.lightgbm;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.BeforeClass;
import org.junit.Test;

import com.feedzai.openml.data.Dataset;
import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.mocks.MockDataset;
import com.feedzai.openml.provider.descriptor.ModelParameter;
import com.feedzai.openml.provider.descriptor.fieldtype.ChoiceFieldType;
import com.feedzai.openml.provider.exception.ModelLoadingException;

import static com.feedzai.openml.provider.lightgbm.FairGBMDescriptorUtil.CONSTRAINT_GROUP_COLUMN_PARAMETER_NAME;
import static com.feedzai.openml.provider.lightgbm.LightGBMBinaryClassificationModelTrainerTest.average;
import static com.feedzai.openml.provider.lightgbm.LightGBMBinaryClassificationModelTrainerTest.ensureFeatureContributions;
import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.BOOSTING_TYPE_PARAMETER_NAME;
import static com.feedzai.openml.provider.lightgbm.LightGBMDescriptorUtil.NUM_ITERATIONS_PARAMETER_NAME;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Tests for the descriptor util classes (both FairGBM and LightGBM).
 *
 * @author Andre Cruz (andre.cruz@feedzai.com)
 * @since 1.4.0
 */
public class DescriptorUtilTest {

    @Test
    public void FairGBMDoesNotSupportRFBoostingType() {
        ChoiceFieldType boostingTypeParamType = (ChoiceFieldType) FairGBMDescriptorUtil.PARAMS
                .stream().filter(el -> el.getName().equals(BOOSTING_TYPE_PARAMETER_NAME))
                .findFirst().orElse(null).getFieldType();

        // Assert an error was returned
        assertThat(boostingTypeParamType.validate(BOOSTING_TYPE_PARAMETER_NAME, "rf"))
                .as("FairGBM boosting type validation error").isPresent();
    }

    @Test
    public void LightGBMSupportsRFBoostingType() {
        ChoiceFieldType boostingTypeParamType = (ChoiceFieldType) LightGBMDescriptorUtil.PARAMS
                .stream().filter(el -> el.getName().equals(BOOSTING_TYPE_PARAMETER_NAME))
                .findFirst().orElse(null).getFieldType();

        // Assert no error was returned
        assertThat(boostingTypeParamType.validate(BOOSTING_TYPE_PARAMETER_NAME, "rf"))
                .as("LightGBM boosting type validation passes").isNotPresent();
    }

}
