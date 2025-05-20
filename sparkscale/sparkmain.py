#starting a spark session

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkML-Practice") \
    .getOrCreate()

df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)
df.show(5)