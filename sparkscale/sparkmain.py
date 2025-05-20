#starting a spark session

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer


spark = SparkSession.builder \
    .appName("SparkML-Practice") \
    .getOrCreate()

df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)
df.show(5)

indexer = StringIndexer(inputCol="M/F", outputCol="gender_index")
df = indexer.fit(df).transform(df)

assembler = VectorAssembler(
    inputCols=["Age", "Visit", "EDUC", "gender_index"], 
    outputCol="features"
)



data = assembler.transform(df).select("features", "Group").show()
