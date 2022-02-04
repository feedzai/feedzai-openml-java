# Microsoft LightGBM Provider

This implementation is a "wrapper" to Microsoft's LightGBM framework for Java focused in performance for real-time environments.

The main goals of the implementation are:

1.  Reliability
2.  Low latency single-event scoring meant for realtime usage
3.  High throughput scoring
4.  High throughput train

## Installation

This module depends on other parts of `feedzai-openml-java`. At the repo root run:

```bash
mvn clean install # or `mvn clean package` to generate the .jar alone
```

With the whole project built, you can now build this module "alone". Use the same command in this folder.


### Build time

The first build is lengthier (might take up to 15 minutes) as it needs to create [make-lightgbm](https://github.com/feedzai/make-lightgbm/)'s docker image and fetch maven dependencies. After that each build takes around 10-30 seconds.

## LightGBM dataset
The [training dataset](lightgbm-provider/src/test/resources/treeshap_t/treeshap_train.csv) is generated based on
the following python code:
```python
np.random.seed(8000)
card = np.random.randint(low=0, high=101, size=50000)
amount = np.random.randint(low=0, high=1000, size=50000)
cat1_generator = np.random.randint(low=0, high=4, size=50000)
cat2_generator = np.random.randint(low=0, high=3, size=50000)
cat3_generator = np.random.randint(low=0, high=6, size=50000)
num1_float = np.random.randint(low=1, high=1001, size=50000)+0.4
num2_float = np.random.randint(low=200000, high=40000001, size=50000) -0.3
num3_float = np.random.randint(low=10000, high=12001, size=50000) -0.1

data = np.array([card, amount, cat1_generator, cat2_generator, cat3_generator, num1_float, num2_float, num3_float])
headers = ["card", "amount","cat1_generator", "cat2_generator","cat3_generator", "num1_float", "num2_float", "num3_float"]
data_df = pd.DataFrame(data.T, columns=headers)
data_df['fraud_label'] = data_df.apply(lambda x : x['amount'] > 400 and x['amount'] < 700  and x["cat1_generator"]==2 and x["num1_float"] < 700, axis=1).astype(int)
```

To summarize, we have the following rules for each generated field:
```
amount          RANDBETWEEN(0,1000)
card	        RANDBETWEEN(0,100)
cat1_generator	RANDBETWEEN(0,3)
cat2_generator	RANDBETWEEN(0,2)
cat3_generator	RANDBETWEEN(0,5)
num1_float  	RANDBETWEEN(1,1000)+0.4
num2_float  	RANDBETWEEN(200000,40000000)-0.3
num3_float  	RANDBETWEEN(10000,12000)
fraud_label     400<amount<700 & cat1_generator="2" & num1_float < 700
```
The [results](lightgbm-provider/src/test/resources/treeshap_t/treeshap_result.csv) were produced using
[Python API v3.3.2.99](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier.predict)
with the following parameters:
```
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True,
    "seed": 42,
    "num_iterations": 100
}
```
Finally, the implementation is ensured based on given input dataset and expected results using
[LightGBMResultTest](lightgbm-provider/src/test/java/com/feedzai/openml/provider/lightgbm/LightGBMResultTest.java).
