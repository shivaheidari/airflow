from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName("sparksvm").getOrCreate()

# Load data
df = spark.read.csv("Data/dementia_dataset.csv", header=True, inferSchema=True)

# Index categorical columns
categorical_col = ["M/F", "Hand", "Group"]
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_col]
indexer_pipeline = Pipeline(stages=indexers)
df_with_index = indexer_pipeline.fit(df).transform(df)

# Filter for binary classification (Group_index == 0.0 or 1.0)
df_binary = df_with_index.filter(df_with_index["Group_index"].isin([0.0, 1.0]))
df_binary = df_binary.filter(df_binary["Group_index"].isNotNull())
df_binary.select("Group_index").distinct().show()

# Assemble features
assembler = VectorAssembler(
    inputCols=["Age"] + [f"{col}_index" for col in categorical_col],
    outputCol="features"
)

# Define SVM model
svm_linear = LinearSVC(
    featuresCol="features",
    labelCol="Group_index",
    maxIter=10
)

# Build pipeline for assembler and SVM
pipeline = Pipeline(stages=[assembler, svm_linear])

# Evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="Group_index",
    predictionCol="prediction",
    metricName="accuracy"
)

# Cross-validation setup
paramGrid = ParamGridBuilder().addGrid(svm_linear.regParam, [0.01, 0.02]).build()
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

print("Data count:", df_binary.count())
print("Starting cross-validation...")

cvModel = cv.fit(df_binary)
print("Cross-validation finished.")

predictions = cvModel.transform(df_binary)
accuracy = evaluator.evaluate(predictions)
print("best model Accuracy", accuracy)