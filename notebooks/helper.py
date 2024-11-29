# -*- coding: utf-8 -*-
"""Compute metrics for Novartis Datathon 2024.
   This auxiliar file is intended to be used by participants in case
   you want to test the metric with your own train/validation splits."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def _CYME(df: pd.DataFrame) -> float:
    """ Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))"""

    yearly_agg = df.groupby("cluster_nl")[["target", "prediction"]].sum().reset_index()
    yearly_error = abs((yearly_agg["target"] - yearly_agg["prediction"])/yearly_agg["target"]).median()

    monthly_error = abs((df["target"] - df["prediction"])/df["target"]).median()

    return 1/2*(yearly_error + monthly_error)


def _metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """Compute metric of submission.

    :param df: Dataframe with target and 'prediction', and identifiers.
    :return: Performance metric
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Split 0 actuals - rest
    zeros = df[df["zero_actuals"] == 1]
    recent = df[df["zero_actuals"] == 0]

    # weight for each group
    zeros_weight = len(zeros)/len(df)
    recent_weight = 1 - zeros_weight

    # Compute CYME for each group
    return [recent_weight*_CYME(recent),  zeros_weight*min(1,_CYME(zeros))]


def compute_metric(submission: pd.DataFrame) -> float:
    """Compute metric.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Performance metric.
    """

    submission["date"] = pd.to_datetime(submission["date"])
    submission = submission[['cluster_nl', 'date', 'target', 'prediction', 'zero_actuals']]

    return round(sum(_metrics(submission)), 8)


def unaveraged_CYME(metric_df: pd.DataFrame, prediction: np.ndarray ) -> float:
    """ Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))"""
    metric_df["prediction"] = prediction
    return _CYME(metric_df)
