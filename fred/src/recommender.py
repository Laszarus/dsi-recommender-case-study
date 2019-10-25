import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructField, StructType
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import time


def top_x(n):
    top_choices = []
    for place in range(1, n+1):
        if place == 1:
            choice = '1st'
        elif place == 2:
            choice = '2nd'
        elif place == 3:
            choice = '3rd'
        else:
            choice = str(place) + 'th'
        top_choices.append(choice)
    return top_choices


def print_user_recommendations(df, userId):
    recs = df[df['userId'] == userId]
    [print(f"{c}: {recs[c].values[0]}") for c in recs.columns]


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__}:  {(te - ts) * 1000 : 2.2} ms')
        return result
    return timed


@timeit
def fit_model(model, data):
    return model.fit(data)


@timeit
def get_predictions(model, data):
    return model.transform(data)


@timeit
def create_spark_df(df):
    return spark.createDataFrame(df)


@timeit
def read_spark_df(filename, sep=',', header=None):
    lines = spark.read.text(filename).rdd
    if header:
        header = lines.first()
        lines = lines.filter(lambda line: line != header)
    parts = lines.map(lambda row: row.value.split(sep))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]),
                                         movieId=int(p[1]),
                                         rating=float(p[2]),
                                         timestamp=int(p[3])))
    schema = StructType([StructField('userId', IntegerType(), False),
                         StructField('movieId', IntegerType(), False),
                         StructField('rating', FloatType(), True),
                         StructField('timestamp', IntegerType(), True)])
    return spark.createDataFrame(ratingsRDD, schema).drop('timestamp')


@timeit
def test_train_split(sdf, split=[0.8, 0.2], seed=42):
    return sdf.randomSplit(split, seed=seed)


@timeit
def get_rmse(sdf):
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    return evaluator.evaluate(sdf)


@timeit
def get_all_user_recommendations(ratings, n):
    return ratings.recommendForAllUsers(n)


if __name__ == "__main__":

    spark = SparkSession.builder.getOrCreate()

    # movies = pd.read_csv('data/movies/movies.csv')
    # pd_ratings = pd.read_csv('data/movies/ratings.csv')
    # pd_ratings = pd_ratings.drop('timestamp', axis=1)
    # spark_ratings = create_spark_df(pd_ratings)

    # Read 100k dataset directly into spark RDD
    # filename = 'data/movies/ratings.csv'
    # spark_ratings = read_spark_df(filename, sep=',', header=True)

    # Read 1M dataset directly into spark RDD
    ratings_fname = 'data/movies-1M/ratings.dat'
    spark_ratings = read_spark_df(ratings_fname, sep='::', header=False)
    movies_fname = 'data/movies-1M/movies.dat'
    pd_movies = pd.read_csv(movies_fname, sep='::', header=None,
                            engine='python')
    pd_movies.columns = ['movieId', 'title', 'genre']
    movie_dict = dict(zip(pd_movies['movieId'], pd_movies['title']))

    train, test = test_train_split(spark_ratings)

    factor_model = ALS(
        itemCol='movieId',
        userCol='userId',
        ratingCol='rating',
        nonnegative=True,
        regParam=0.1,
        coldStartStrategy='drop',
        rank=20)

    ratings = fit_model(factor_model, train)
    predict = get_predictions(ratings, test)
    rmse = get_rmse(predict)

    n = 5
    userRecs = get_all_user_recommendations(ratings, n)
    # movieRecs = ratings.recommendForAllItems(n)

    best_movies = userRecs.toPandas()
    ranked = pd.DataFrame(best_movies['recommendations'].tolist(),
                          index=best_movies.index)
    ranked.columns = top_x(n)

    for col in ranked.columns:
        data = ranked[col]
        # ranked[col+'_rating'] = data.map(lambda x: np.round(x[1], 2))
        ranked[col] = data.map(lambda x: movie_dict.get(x[0], 0))

    best_movies = best_movies.join(ranked)
    best_movies.drop('recommendations', inplace=True, axis=1)

    print_user_recommendations(best_movies, 4900)
    
