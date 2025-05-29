from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder \
    .appName("SparkML-Practice") \
    .getOrCreate()

print("Spark UI URL (while running):", spark.sparkContext.uiWebUrl)  # Check URL

df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)
df.show(5)

# --- Your ML Pipeline ---
categorical_cols = ["M/F", "Hand", "Group"]
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_cols]
assembler = VectorAssembler(inputCols=["Age"] + [f"{col}_index" for col in categorical_cols], outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="Group_index")

pipeline = Pipeline(stages=indexers + [assembler, lr])
model = pipeline.fit(df)  # Triggers DAG execution

# Print the physical plan (DAG)
print("Execution plan:")
df._jdf.queryExecution().executedPlan().toString()  # Detailed DAG

# Keep the session alive to access UI
input("Press Enter to exit and close Spark UI...")  # Now open localhost:4040 in your browser
spark.stop()
