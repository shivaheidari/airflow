from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

#spark session
spark = SparkSession.builder \
    .appName("SparkML-Practice") \
    .getOrCreate()


#load data
df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)

print(df.columns)
categorical_cols = ["M/F", "Hand", "Group"]
indexers = [StringIndexer(inputCols=col, outputCol=f"{col}_index") for col in categorical_cols]
assembeler = VectorAssembler(inputCols=["Age"]+[f"{col}_indexe" for col in categorical_cols], outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="Group_index")

pipeline = Pipeline(stages=indexers + [assembeler, lr])
model = pipeline.fit(df)

