
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression


spark = SparkSession.builder \
    .appName("SparkML-Practice") \
    .getOrCreate()


df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)
df.show(5)

categorical_cols = ["M/F", "Hand", "Group"]

df_filtered = df.filter(df["AGE"] > 30)
df_grouped = df_filtered.groupBy("department").count()

# Print the logical/physical plan (text-based)
print(df_grouped.explain(extended=True))  # Shows parsed/logical/optimized/physical plans



indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in categorical_cols]


pipeline = Pipeline(stages=indexers)
indexed_df = pipeline.fit(df).transform(df)
train , test = indexed_df.randomSplit([0.8, 0.2], seed=42)
assembler = VectorAssembler(
    inputCols=["Age", "Visit", "EDUC", "M/F_index", "Hand_index", "ASF", "nWBV"], 
    outputCol="features"
)



lr = LogisticRegression(featuresCol="features", labelCol="Group_index")

pipeline = Pipeline(stages=[assembler, lr])

model = pipeline.fit(train)

predictions = model.transform(test)
predictions.select("features", "Group_index", "prediction", "probability").show()


