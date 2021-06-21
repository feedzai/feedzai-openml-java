package com.feedzai.openml.h2o.algos;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.feedzai.openml.data.schema.DatasetSchema;
import com.feedzai.openml.h2o.H2OAlgorithm;
import com.feedzai.openml.h2o.server.H2OApp;
import com.feedzai.openml.provider.descriptor.MLAlgorithmDescriptor;
import com.feedzai.openml.util.algorithm.MLAlgorithmEnum;

/**
 * Factory class responsible for providing the correct H20 Algorithm util.
 *
 * @author Antonio Silva (antonio.silva@feedzai.com)
 * @since @@@feedzai.next.release@@@
 */
public class H2OAlgoUtilsFactory {

    /**
     * Logger for this class.
     */
    private static final Logger logger = LoggerFactory.getLogger(H2OApp.class);

    /**
     * Private constructor to avoid class instantiation.
     *
     * @since @@@feedzai.next.release@@@
     */
    private H2OAlgoUtilsFactory() {
    }

    /**
     * Factory main method to get the correct H20 Algo Utils.
     * @param algorithmDescriptor descriptor for the algorithm we want the utils for.
     * @param schema              datasource schema.
     * @return AbstractH2OAlgoUtils implementation for the algoDescriptor/schema provided.
     *
     * @since @@@feedzai.next.release@@@
     */
    public static AbstractH2OAlgoUtils getH2OAlgoUtils(final MLAlgorithmDescriptor algorithmDescriptor,
                                                       final DatasetSchema schema) {
        //TODO: the AbstractH2OAlgoUtils objects have no state, create one for each call or have singletons to return ?
        switch (getH2OAlgorithm(algorithmDescriptor)) {
            case DISTRIBUTED_RANDOM_FOREST:
                return new H2ODrfUtils();
            case XG_BOOST:
                return new H2OXgboostUtils();
            case DEEP_LEARNING:
                return new H2ODeepLearningUtils();
            case GRADIENT_BOOSTING_MACHINE:
                return new H2OGbmUtils();
            case NAIVE_BAYES_CLASSIFIER:
                return new H2OBayesUtils();
            case GENERALIZED_LINEAR_MODEL:
                return new H2OGeneralizedLinearModelUtils(schema);
            case ISOLATION_FOREST:
                return new H2OIsolationForestUtils();
            default:
                final String errorMessage = String.format("Unsupported algorithm: %s", algorithmDescriptor.getAlgorithmName());
                logger.error(errorMessage);
                throw new IllegalArgumentException(errorMessage);
        }
    }

    /**
     * Resolves the H2O algorithm from the provided descriptor.
     *
     * @param algorithmDescriptor The algorithm descriptor from which the {@link H2OAlgorithm} is resolved.
     * @return The resolve {@link H2OAlgorithm}.
     *
     * @since @@@feedzai.next.release@@@
     */
    private static H2OAlgorithm getH2OAlgorithm(final MLAlgorithmDescriptor algorithmDescriptor) {
        return MLAlgorithmEnum.getByName(H2OAlgorithm.values(), algorithmDescriptor.getAlgorithmName())
                              .orElseThrow(() -> new IllegalArgumentException("Unknown algorithm: " + algorithmDescriptor));
    }
}
