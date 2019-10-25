from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.dataframe import DataFrame
import pandas as pd
import numpy as np


def spark_shape(self):
    return (self.count(), len(self.columns))


DataFrame.shape = spark_shape


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    df = pd.read_csv('data/u.data', sep='\t', header=None,
                     names=['user', 'movie', 'rating', 'timestamp'])

    df.drop(columns='timestamp', inplace=True)
    sdf = spark.createDataFrame(df)