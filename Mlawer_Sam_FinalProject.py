from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
import math

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


if __name__ == "__main__":
    spark = SparkSession.builder.appName("FinalProject").getOrCreate()

    df = spark.read.csv("Covid Data.csv", header=True, inferSchema=True)
    # df.show()
    # df.printSchema()

    # Data Cleaning
    df = df.filter(df["CLASIFFICATION_FINAL"] <= 3)

    df = df.withColumn(
        "died",
        when(col("DATE_DIED") == "9999-99-99", 0).otherwise(1)
    )

    df = df.drop("USMER","MEDICAL_UNIT","PATIENT_TYPE","INTUBED","CLASIFFICATION_FINAL","ICU","DATE_DIED")

    cols_to_update = ["PNEUMONIA", "PREGNANT","DIABETES","COPD","ASTHMA","INMSUPR","HIPERTENSION","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO"]
    target_values = [97,98,99]
    new_value = 1

    for col_name in cols_to_update:
        df = df.withColumn(
            col_name,
            when(col(col_name).isin(target_values), new_value).otherwise(col(col_name))
    )
        
    # df.show()

    # Balance classes
    df_majority = df.filter(col("died") == 0)
    df_minority = df.filter(col("died") == 1)

    minority_count = df_minority.count()
    majority_count = df_majority.count()
    fraction = minority_count / majority_count
    df_majority_downsampled = df_majority.sample(withReplacement=False, fraction=fraction, seed=42)
    df_balanced = df_majority_downsampled.union(df_minority)
 

    # Build Logistic Regression
    assembler = VectorAssembler(
        inputCols=["SEX","PNEUMONIA", "AGE", "PREGNANT","DIABETES","COPD","ASTHMA","INMSUPR","HIPERTENSION","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO"],
        outputCol="features"
    )
    df_balanced = assembler.transform(df_balanced)
    train_df, test_df = df_balanced.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()

    lr = LogisticRegression(featuresCol="features", labelCol="died")
    lr_model = lr.fit(train_df)
    predictions = lr_model.transform(test_df)
    predictionAndLabels = predictions.select("prediction", "died") \
        .rdd.map(lambda row: (float(row['prediction']), float(row['died'])))
    
    metrics = MulticlassMetrics(predictionAndLabels)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="died", predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator.evaluate(predictions)
    print("F1 score:", f1)

    print("\nPerformance Metrics: Logistic Regression")
    print("Precision:", metrics.precision(1.0))
    print("Recall:", metrics.recall(1.0))
    print("F1:", metrics.fMeasure(1.0))
    print(f"Accuracy = {metrics.accuracy}")
    print("Confusion Matrix:", metrics.confusionMatrix())


    # Build random forest
    rf = RandomForestClassifier(featuresCol="features", labelCol="died", numTrees=100)
    rf_model = rf.fit(train_df)
    predictions = rf_model.transform(test_df)

    predictionAndLabels = predictions.select("prediction", "died") \
        .rdd.map(lambda row: (float(row['prediction']), float(row['died'])))
    
    metrics = MulticlassMetrics(predictionAndLabels)

    print("\nPerformance Metrics: Random Forest")
    print("Precision:", metrics.precision(1.0))
    print("Recall:", metrics.recall(1.0))
    print("F1:", metrics.fMeasure(1.0))
    print(f"Accuracy = {metrics.accuracy}")
    print("Confusion Matrix:", metrics.confusionMatrix())



    # Build GBT
    gbt = GBTClassifier(featuresCol="features", labelCol="died", maxIter=100)
    gbt_model = gbt.fit(train_df)
    predictions = gbt_model.transform(test_df)
    predictionAndLabels = predictions.select("prediction", "died") \
        .rdd.map(lambda row: (float(row['prediction']), float(row['died'])))
    
    metrics = MulticlassMetrics(predictionAndLabels)

    print("\nPerformance Metrics: GBT")
    print("Precision:", metrics.precision(1.0))
    print("Recall:", metrics.recall(1.0))
    print("F1:", metrics.fMeasure(1.0))
    print(f"Accuracy = {metrics.accuracy}")
    print("Confusion Matrix:", metrics.confusionMatrix())


    # Build Naive-Bayes
    nb = NaiveBayes(featuresCol="features", labelCol="died", smoothing=1.0, modelType="multinomial")
    nb_model = nb.fit(train_df)
    predictions = nb_model.transform(test_df)
    predictionAndLabels = predictions.select("prediction", "died") \
        .rdd.map(lambda row: (float(row['prediction']), float(row['died'])))
    
    metrics = MulticlassMetrics(predictionAndLabels)

    print("\nPerformance Metrics: Naive Bayes")
    print("Precision:", metrics.precision(1.0))
    print("Recall:", metrics.recall(1.0))
    print("F1:", metrics.fMeasure(1.0))
    print(f"Accuracy = {metrics.accuracy}")
    print("Confusion Matrix:", metrics.confusionMatrix())


    spark.stop()