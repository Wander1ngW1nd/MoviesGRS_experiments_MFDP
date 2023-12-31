import numpy as np
import pandas as pd

np.random.seed(42)


DATA_PATH = "../data/"


def fetch_train_test_data(groups_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data: pd.DataFrame = pd.read_parquet(DATA_PATH + "ratings_train.pq")
    test_data: pd.DataFrame = pd.read_parquet(DATA_PATH + "ratings_test.pq")

    test_data["group1"] = test_data.userId
    for i, group in enumerate(groups_list[1:]):
        group_assignment: pd.DataFrame = pd.read_parquet(DATA_PATH + f"{group}.pq")
        test_data = test_data.merge(group_assignment, on="userId").rename(columns={"group": f"group{i+2}"})

    return train_data, test_data


def load_movies_data() -> pd.DataFrame:
    movies_data: pd.DataFrame = pd.read_parquet(DATA_PATH + "movies_train.pq")
    return movies_data


def compute_users_unwatched_movies(ratings: pd.DataFrame) -> pd.DataFrame:
    top_popular_movies: pd.DataFrame = (
        ratings.groupby(by="movieId")
        .agg({"userId": "nunique"})
        .sort_values(by="userId", ascending=False)
        .rename(columns={"userId": "userCount"})
        .reset_index()
    )

    movie_ids: np.array = top_popular_movies.movieId.values

    unwatched_movies: pd.DataFrame = ratings.groupby(by="userId").agg({"movieId": list}).reset_index()
    unwatched_movies["unwatched"] = unwatched_movies.movieId.apply(
        lambda x: movie_ids[np.isin(movie_ids, x, invert=True)]
    )

    unwatched_movies = unwatched_movies[["userId", "unwatched"]]

    return unwatched_movies


def collect_users_watch_history(data: pd.DataFrame, groups_list: list[str]) -> pd.DataFrame:
    users_watch_history: pd.DataFrame = (
        data.sort_values(by="rating", ascending=False)
        .groupby(by="userId")
        .agg({**{g: "first" for g in groups_list}, "movieId": list, "rating": list})  # type: ignore [misc]
        .reset_index()
    )
    users_watch_history["movieId"] = users_watch_history.movieId.apply(np.array)
    users_watch_history["rating"] = users_watch_history.rating.apply(np.array)

    return users_watch_history


def collect_unwatched_train_watched_test_movies(groups_list: list[str]) -> pd.DataFrame:
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    train_data, test_data = fetch_train_test_data(groups_list)

    users_unwatched_movies_train: pd.DataFrame = compute_users_unwatched_movies(train_data)
    users_watch_history_test: pd.DataFrame = collect_users_watch_history(test_data, groups_list)
    unwatched_train_watched_test_ratings_test: pd.DataFrame = users_watch_history_test.merge(
        users_unwatched_movies_train, on=["userId"]
    )

    return unwatched_train_watched_test_ratings_test
