# H2O API

This module contains the classes and logic related with the import of a model generated in H2O.

The current implementation allows to import a model generated in one of the following two formats:
* Plain Old Java Object - [POJO](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html#pojo-quick-start)
* Model Object, Optimized - [MOJO](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html#mojo-quick-start)

## POJO

H2O allows you to convert the models you have built to a Plain Old Java Object (POJO). POJOs allow users to build a model using H2O and then deploy the model to score in real-time.

**Note**: POJOs are not supported for XGBoost, Stacked Ensembles, or AutoML models.

#### Format File

The generation of a POJO in H2O creates a Java file with the model.
**The exported Java file needs to be compiled into a Jar to be used aftwards.**

For that you need to download the 'h2o-genmodel.jar' that is available in [h2o.ai](https://www.h2o.ai) and execute:
```
javac -cp h2o-genmodel.jar -J-Xmx2g -J-XX:MaxPermSize=128m <generated_model>.java 
jar cvfe <generated_model>.jar <generated_model> *.class
```

Afterwards, you can use the file created with the command above (<generated_model>.jar) to import the model.

Note: OpenML assumes that the JAR file is named with the name of the Main Class of the file, and so you might have troubles importing the model if you have previously changed the filename.

## MOJO

A MOJO (Model Object, Optimized) is an alternative to H2O's currently available POJO. As with POJOs, H2O allows you to convert models that you build to MOJOs, which can then be deployed for scoring in real time.

Note: MOJOs are supported for Deep Learning, DRF, GBM, GLM, GLRM, K-Means, Stacked Ensembles, SVM, Word2vec, and XGBoost algorithms.

#### Format File

The generation of a MOJO in H2O creates a ZIP file with the model. 
You can use that ZIP file to import the model.

## Model File Structure

To import POJOs (compiled into jars) or MOJOs you need to place them in a special folder structure (and optionally provide a schema).

```
<model name>/
├── model
│   └── <generated_model>.zip or <generated_model>.jar 
└── model.json
```

\<model name\> can be any name you want to give to the folder containing the model. You will need to provide the path to this folder.

"model.json" is an optional JSON file describing the schema of the model. It needs to be in the following format:

```
{
   "targetIndex":2,
   "fieldSchemas":[
      {
         "fieldIndex":0,
         "fieldName":"date",
         "valueSchema":{
            "@type":"numeric",
            "allowMissing":false
         }
      },
      {
         "fieldIndex":1,
         "fieldName":"Field names can have spaces and symbols!",
         "valueSchema":{
            "@type":"numeric",
            "allowMissing":true
         }
      },
      {
         "fieldIndex":2,
         "fieldName":"fraud",
         "valueSchema":{
            "@type":"categorical",
            "allowMissing":false,
            "nominalValues":[
               "false",
               "true"
            ]
         }
      }
   ]
} 
```

This is an example of a schema definition for a model. 
If you don't provide a "model.json" file for a model you will need to create the schema and add it through the model import UI.

Notes:
- The field indices ("fieldIndex") for fieldSchemas are required and start at 0.
- Field schemas must be ordered by ascending field index.
- The target index ("targetIndex") must be a valid index pointing to the field schema of the target variable of the model.
- The available value schema types for H2O models are, as shown, "categorical" and "numeric".
- "allowMissing" is a boolean value that indicates whether a field can be missing in a valid instance.
- In a categorical value schema, the possible categories (nominal values) are listed in an array on the "nominalValues" property.
