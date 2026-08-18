"""Prediction models. LightGBM is the baseline; add new models behind the same interface."""

from models.lightgbm_model import LightGBMModel

__all__ = ["LightGBMModel"]
