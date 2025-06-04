from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("sparksvm").getOrCreate()

df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)
df.show()