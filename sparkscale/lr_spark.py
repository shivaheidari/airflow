from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

#spark session
spark = SparkSession.builder \
    .appName("SparkML-LR") \
    .getOrCreate()


#load data
df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)

print(df.columns)
categorical_cols = ["M/F", "Hand", "Group"]
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_cols]
assembler = VectorAssembler(inputCols=["Age"]+[f"{col}_index" for col in categorical_cols], outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="Group_index"
                        ,maxIter=100,
                        regParam=0.01, 
                        elasticNetParam=0.0, 
                        tol=1e-6,
                        fitIntercept=True,
                        standardization=True)

pipeline = Pipeline(stages=indexers + [assembler, lr])
model = pipeline.fit(df)

predictions = model.transform(df)
evaluation_acc = MulticlassClassificationEvaluator(labelCol="Group_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluation_acc.evaluate(predictions)
print("Accuracy", accuracy)

evaluation_f1 = MulticlassClassificationEvaluator(labelCol="Group_index", predictionCol="prediction", metricName="f1")
f1 = evaluation_f1.evaluate(predictions)
print("f1", f1)

