#starting a spark session

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder \
    .appName("SparkML-Practice") \
    .getOrCreate()

df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)
df.show(5)
categorical_cols = ["M/F", "Hand"]


indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in categorical_cols]



assembler = VectorAssembler(
    inputCols=["Age", "Visit", "EDUC", "M/F_index", "Hand_index"], 
    outputCol="features"
)

lr = LogisticRegression(featuresCol="features", labelCol="Group")

pipeline = Pipeline(stages=[indexers, assembler, lr])

