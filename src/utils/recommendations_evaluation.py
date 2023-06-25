from functools import reduce
from typing import Callable

import numpy as np
import pandas as pd


def generate_recommendations(
    recommendation_function: Callable[[pd.Series], pd.Series],
    unwatched_train_watched_test_ratings_test: pd.DataFrame,
    groups_names: list[str],
) -> pd.DataFrame:
    """
    Gets recommendation function; dataset containing movies unwatched in train by each user in train and
    movies watched in test; list of groups' names.
    Returns the provided dataset (unwatched_train_watched_test_movies) merged with group recommendations
    that each user received in every group she was allocated to.
    """

    for group in groups_names:
        # collect sets of group members and movies not watched by everyone in the group
        group_unwatched_movies: pd.DataFrame = (
            unwatched_train_watched_test_ratings_test.groupby(by=group)
            .agg({"userId": list, "unwatched": lambda x: np.array(reduce(np.intersect1d, x))})
            .reset_index()
        )
        group_unwatched_movies["userId"] = group_unwatched_movies.userId.apply(np.array)

        # generate recommendations for group
        group_unwatched_movies[f"{group}_rec"] = group_unwatched_movies.apply(recommendation_function, axis=1)

        # merge group recommendations with original user-wise data
        unwatched_train_watched_test_ratings_test = unwatched_train_watched_test_ratings_test.merge(
            group_unwatched_movies[[group, f"{group}_rec"]], on=group
        )

    return unwatched_train_watched_test_ratings_test


def collect_relevance_of_recommendations(
    unwatched_train_watched_test_ratings_test: pd.DataFrame, group: str
) -> pd.DataFrame:
    """
    Gets dataframe containig all users test watch history.
    Returns the same dataframe with column of relevance arrays. Relevance arrays contain 1
    if the movie was presented in test watch history and 0 otherwise, for every recommended movie.
    """
    unwatched_train_watched_test_ratings_test[f"{group}_relevance"] = unwatched_train_watched_test_ratings_test.apply(
        lambda row: np.isin(row[f"{group}_rec"], row["movieId"]).astype(int), axis=1
    )
    return unwatched_train_watched_test_ratings_test


def fetch_user_test_ratings_for_recommended_movies(
    user_unwatched_train_watched_test_ratings_test: pd.Series, group_name: str
) -> np.array:
    """
    Gets user's test watch history and ratings from provided dataframe.
    Returns array of her ratings for movies that were recommended for given group split.
    If the movies was not watched by user in test, it is given 0 rating
    """
    return np.array(
        [
            np.array(user_unwatched_train_watched_test_ratings_test.rating)[
                user_unwatched_train_watched_test_ratings_test.movieId == x
            ].astype(int)[0]
            if np.sum(user_unwatched_train_watched_test_ratings_test.movieId == x)
            else 0
            for x in user_unwatched_train_watched_test_ratings_test[f"{group_name}_rec"]
        ]
    )


def calculate_precision_at_k(unwatched_train_watched_test_ratings_test: pd.DataFrame, group: str) -> pd.DataFrame:
    unwatched_train_watched_test_ratings_test[f"{group}_P_k"] = unwatched_train_watched_test_ratings_test[
        f"{group}_relevance"
    ].apply(lambda relevance: np.cumsum(relevance) * relevance / np.arange(1, len(relevance) + 1), 2)
    unwatched_train_watched_test_ratings_test[f"{group}_P_k"] = unwatched_train_watched_test_ratings_test.apply(
        lambda row: row[f"{group}_P_k"].sum() / min(len(row["movieId"]), len(row[f"{group}_rec"])), axis=1
    )
    unwatched_train_watched_test_ratings_test[f"{group}_P_k"] = unwatched_train_watched_test_ratings_test[
        f"{group}_P_k"
    ].apply(lambda x: np.around(x, 2))
    return unwatched_train_watched_test_ratings_test


def calculate_ndcg(unwatched_train_watched_test_ratings_test: pd.DataFrame, group: str) -> pd.DataFrame:
    unwatched_train_watched_test_ratings_test[f"{group}_rec_ratings"] = unwatched_train_watched_test_ratings_test.apply(
        lambda row: fetch_user_test_ratings_for_recommended_movies(row, group), axis=1
    )

    dcg: np.array = (
        unwatched_train_watched_test_ratings_test[f"{group}_rec_ratings"]
        .apply(lambda ratings: np.sum([rating / np.log2(2 + i) for i, rating in enumerate(ratings)]))
        .values
    )

    idcg: np.array = (
        1e-8
        + unwatched_train_watched_test_ratings_test[f"{group}_rec_ratings"]
        .apply(lambda ratings: np.sum([rating / np.log2(2 + i) for i, rating in enumerate(-np.sort(-ratings))]))
        .values
    )

    unwatched_train_watched_test_ratings_test[f"{group}_NDCG_k"] = dcg / idcg

    return unwatched_train_watched_test_ratings_test


def evaluate_recommendations(
    unwatched_train_watched_test_movies_with_recommendations: pd.DataFrame, groups_names: list[str]
) -> pd.DataFrame:
    metrics_results = {}

    for group in groups_names:
        unwatched_train_watched_test_movies_with_recommendations = collect_relevance_of_recommendations(
            unwatched_train_watched_test_movies_with_recommendations, group
        )

        unwatched_train_watched_test_movies_with_recommendations = calculate_precision_at_k(
            unwatched_train_watched_test_movies_with_recommendations, group
        )

        metrics_name = f"MAP_{group}"
        metrics_value: pd.DataFrame = (
            unwatched_train_watched_test_movies_with_recommendations.groupby(by=group)[f"{group}_P_k"].mean().mean()
        )

        metrics_results[metrics_name] = metrics_value

        unwatched_train_watched_test_movies_with_recommendations = calculate_ndcg(
            unwatched_train_watched_test_movies_with_recommendations, group
        )

        metrics_name = f"NDCG_{group}"
        metrics_value = (
            unwatched_train_watched_test_movies_with_recommendations.groupby(by=group)[f"{group}_NDCG_k"].mean().mean()
        )

        metrics_results[metrics_name] = metrics_value

    metrics_results_2d: dict[str, dict[str, float]] = {}
    for res in metrics_results:  # pylint: disable=consider-using-dict-items
        metrics_name, group = res.split("_")
        metrics_results_2d[metrics_name] = {**metrics_results_2d.get(metrics_name, {}), **{group: metrics_results[res]}}

    return pd.DataFrame(metrics_results_2d)
