from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("sparkML-DT").getOrCreate()

#load data
df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)


Categorical_col = ["M/F", "Hand", "Group"]
result = []
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index")for col in Categorical_col]
assembler = VectorAssembler(inputCols=["Age"]+[f"{col}_index" for col in Categorical_col], outputCol="features")

ds = DecisionTreeClassifier(featuresCol="features", labelCol="Group_index",
                            maxDepth=6,
                            maxBins=32,
                            minInstancesPerNode=1, 
                            impurity="gini")

pipeline = Pipeline(stages=indexers+[assembler, ds])
model = pipeline.fit(df)

predictions = model.transform(df)

evaluation_acc = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="Group_index", metricName="accuracy")
accuracy = evaluation_acc.evaluate(predictions)
print("accuracy", accuracy)

evaluation_f1 = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="Group_index", metricName="f1")
f1 = evaluation_f1.evaluate(predictions)
print("f1", f1)

