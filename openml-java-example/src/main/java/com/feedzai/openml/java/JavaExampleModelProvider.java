package com.feedzai.openml.java;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.provider.descriptor.fieldtype.ParamValidationError;
import com.feedzai.openml.provider.exception.ModelLoadingException;
import com.feedzai.openml.provider.model.MachineLearningModelLoader;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * FIXME
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 0.1.0
 */
public class JavaExampleModelProvider implements MachineLearningModelLoader<JavaExampleClassificationModel> {

    @Override
    public JavaExampleClassificationModel loadModel(Path path, DatasetSchema datasetSchema) throws ModelLoadingException {
        return null;
    }

    @Override
    public List<ParamValidationError> validateForLoad(Path path, DatasetSchema datasetSchema, Map<String, String> map) {
        return null;
    }

    @Override
    public DatasetSchema loadSchema(Path path) throws ModelLoadingException {
        return null;
    }
}
