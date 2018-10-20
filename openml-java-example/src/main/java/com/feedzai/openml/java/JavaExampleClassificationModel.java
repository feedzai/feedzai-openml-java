package com.feedzai.openml.java;

import com.feedzai.openml.data.Instance;
import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.model.ClassificationMLModel;

import java.nio.file.Path;

/**
 * Example that would have to be filled in.
 *
 * @author Nuno Diegues (nuno.diegues@feedzai.com)
 * @since 0.1.0
 */
public class JavaExampleClassificationModel implements ClassificationMLModel {
    
    @Override
    public double[] getClassDistribution(Instance instance) {
        return new double[0];
    }

    @Override
    public int classify(Instance instance) {
        return 0;
    }

    @Override
    public boolean save(Path path, String s) {
        return false;
    }

    @Override
    public DatasetSchema getSchema() {
        return null;
    }

    @Override
    public void close() throws Exception {

    }
}
