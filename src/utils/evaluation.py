from functools import reduce
from typing import Callable

import numpy as np
import pandas as pd


def generate_recommendations(
    make_recommendations: Callable[[pd.Series], pd.Series], recommender_data: pd.DataFrame, groups_list: list[str]
) -> pd.DataFrame:
    for group in groups_list:
        group_unwatched: pd.DataFrame = (
            recommender_data.groupby(by=group)
            .agg({"userId": list, "unwatched": lambda x: np.array(reduce(np.intersect1d, x))})
            .reset_index()
        )
        group_unwatched["userId"] = group_unwatched.userId.apply(np.array)
        group_unwatched[f"{group}_rec"] = group_unwatched.apply(make_recommendations, axis=1)

        recommender_data = recommender_data.merge(group_unwatched[[group, f"{group}_rec"]], on=group)

    return recommender_data


def get_rating(row: pd.Series, group: str) -> np.array:
    return np.array(
        [
            np.array(row.rating)[row.movieId == x].astype(int)[0] if np.sum(row.movieId == x) else 0
            for x in row[f"{group}_rec"]
        ]
    )


def get_relevance(recommender_data: pd.DataFrame, group: str) -> pd.DataFrame:
    recommender_data[f"{group}_relevance"] = recommender_data.apply(
        lambda row: np.isin(row[f"{group}_rec"], row["movieId"]).astype(int), axis=1
    )
    return recommender_data


def calc_precision_at_k(recommender_data: pd.DataFrame, group: str) -> pd.DataFrame:
    recommender_data[f"{group}_P_k"] = recommender_data[f"{group}_relevance"].apply(
        lambda x: np.cumsum(x) * x / np.arange(1, len(x) + 1), 2
    )
    recommender_data[f"{group}_P_k"] = recommender_data.apply(
        lambda row: row[f"{group}_P_k"].sum() / min(len(row["movieId"]), len(row[f"{group}_rec"])), axis=1
    )
    recommender_data[f"{group}_P_k"] = recommender_data[f"{group}_P_k"].apply(lambda x: np.around(x, 2))
    return recommender_data


def calc_ndcg(recommender_data: pd.DataFrame, group: str) -> pd.DataFrame:
    recommender_data[f"{group}_rec_ratings"] = recommender_data.apply(lambda x: get_rating(x, group), axis=1)

    dcg: np.array = (
        recommender_data[f"{group}_rec_ratings"]
        .apply(lambda ratings: np.sum([r / np.log2(2 + i) for i, r in enumerate(ratings)]))
        .values
    )

    idcg: np.array = (
        1e-8
        + recommender_data[f"{group}_rec_ratings"]
        .apply(lambda ratings: np.sum([r / np.log2(2 + i) for i, r in enumerate(-np.sort(-ratings))]))
        .values
    )

    recommender_data[f"{group}_NDCG_k"] = dcg / idcg

    return recommender_data


def evaluate_recommendations(recommender_data: pd.DataFrame, groups_list: list[str]) -> pd.DataFrame:
    metrics_results = {}

    for group in groups_list:
        recommender_data = get_relevance(recommender_data, group)

        recommender_data = calc_precision_at_k(recommender_data, group)

        metrics_name = f"MAP_{group}"
        metrics_value: pd.DataFrame = recommender_data.groupby(by=group)[f"{group}_P_k"].mean().mean()

        metrics_results[metrics_name] = metrics_value

        recommender_data = calc_ndcg(recommender_data, group)

        metrics_name = f"NDCG_{group}"
        metrics_value = recommender_data.groupby(by=group)[f"{group}_NDCG_k"].mean().mean()

        metrics_results[metrics_name] = metrics_value

    metrics_results_2d: dict[str, dict[str, float]] = {}
    for res in metrics_results:  # pylint: disable=consider-using-dict-items
        metrics_name, group = res.split("_")
        metrics_results_2d[metrics_name] = {**metrics_results_2d.get(metrics_name, {}), **{group: metrics_results[res]}}

    return pd.DataFrame(metrics_results_2d)
