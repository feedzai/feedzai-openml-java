# DataRobot API

This module contains the classes and logic related with the import of a model generated in [DataRobot](https://www.datarobot.com/).

The current implementation allows to import a binary classification model exported in a Jar file.

## Model File Structure

To import DataRobot models you need to place them in a special folder structure (and optionally provide a schema).

```
<model name>/
├── model
│   └── <generated_model>.jar
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
- The available value schema types for DataRobot models are, as shown, "categorical", "string" and "numeric".
- "allowMissing" is a boolean value that indicates whether a field can be missing in a valid instance.
- In a categorical value schema, the possible categories (nominal values) are listed in an array on the "nominalValues" property.
