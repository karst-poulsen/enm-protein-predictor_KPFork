import csv
from dataclasses import dataclass, field
from datetime import datetime as dt
import json
import logging
import pandas as pd
import pathlib
import pickle
from typing import List, Tuple, Union
from yamldataclassconfig.config import YamlDataClassConfig

from . import data_utils, predictor_utils, validation_utils


@dataclass
class Config(YamlDataClassConfig):
    CLEAN_ACCESSION_NUMBERS_FIELDNAME: str = 'Accesion Number'
    CLEAN_CATEGORICAL_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_DROP_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_ENRICHMENT_SPLIT_VALUE: int = None
    CLEAN_FILL_NAN_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_MASK_PATH: str = ''
    CLEAN_MULTI_LABEL_ENCODE_FIELDS: List[str] = field(default_factory=lambda: [])
    CLEAN_TARGET_FIELDNAME: str = None
    INIT_DATA_PATH: str = None
    MODEL_BOOTSTRAP: bool = False
    MODEL_NUM_TREES: int = None
    MODEL_MIN_SAMPLE_SPLIT: int = None
    MODEL_MAX_FEATURES: str = None
    MODEL_MAX_DEPTH: int = None
    MODEL_CRITERION: str = None
    MODEL_NUM_JOBS: int = None
    MODEL_PERSIST: dict = field(default_factory=lambda: {'PERSIST': False})
    OPT_BEST_PARAMS_WRITE_PATH: str = ''
    OPT_NUM_FOLDS: int = 2
    OPT_NUM_TREES_GRID: List[int] = field(default_factory=lambda: [1])
    OPT_MAX_FEATURES_GRID: List[str] = field(default_factory=lambda: ['auto'])
    OPT_MAX_DEPTH_GRID: List[int] = field(default_factory=lambda: [2])
    OPT_MIN_SAMPLES_SPLIT_GRID: List[int] = field(default_factory=lambda: [2])
    OPT_CRITERION_GRID: List[str] = field(default_factory=lambda: ['gini'])
    RFECV_MASK_WRITE_PATH: str = ''
    RFECV_MIN_FEATURES: int = 1
    RFECV_NUM_FOLDS: int = 5
    RFECV_SCORING: str = 'f1'
    RFECV_STEP: float = 1
    TRAIN_METRICS_WRITE_PATH: str = ''
    TRAIN_SPLIT_PERCENTAGE: float = 0.6
    VALIDATION_PROBABLITY_THRESHOLD: float = 0.5

    def list_variables(self):
        return [self.CLEAN_ACCESSION_NUMBERS_FIELDNAME, self.CLEAN_CATEGORICAL_FIELDS,
                self.CLEAN_DROP_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE,
                self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_MASK_PATH, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS,
                self.CLEAN_TARGET_FIELDNAME, self.INIT_DATA_PATH, self.MODEL_BOOTSTRAP, self.MODEL_NUM_TREES,
                self.MODEL_MIN_SAMPLE_SPLIT, self.MODEL_MAX_FEATURES, self.MODEL_MAX_DEPTH,
                self.MODEL_CRITERION, self.MODEL_NUM_JOBS, self.MODEL_PERSIST, self.OPT_BEST_PARAMS_WRITE_PATH,
                self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID,
                self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID, self.OPT_MIN_SAMPLES_SPLIT_GRID,
                self.OPT_CRITERION_GRID, self.RFECV_MASK_WRITE_PATH, self.RFECV_MIN_FEATURES, self.RFECV_NUM_FOLDS,
                self.RFECV_SCORING, self.RFECV_STEP, self.TRAIN_METRICS_WRITE_PATH, self.TRAIN_SPLIT_PERCENTAGE,
                self.VALIDATION_PROBABLITY_THRESHOLD]


class Pipeline:
    """
    A Pipeline is a convenience tool that allows the user to easily optimize and train a model on cleaned
    data.

    Args:
        config_path: str | Optional parameter to specify config file for the pipeline.

    """
    def __init__(self, config_path="config/config-train.yml"):
        self.PARENT_ABSOLUTE_PATH = pathlib.Path(pathlib.Path().parent.absolute()).parent.absolute()
        self.CONFIGS = Config()
        self.CONFIG_PATH = self.get_absolute_path(config_path)
        self.CONFIGS.load(self.CONFIG_PATH)
        self.CLEAN_ACCESSION_NUMBERS_FIELDNAME, self.CLEAN_CATEGORICAL_FIELDS, self.CLEAN_DROP_FIELDS, self.CLEAN_ENRICHMENT_SPLIT_VALUE, self.CLEAN_FILL_NAN_FIELDS, self.CLEAN_MASK_RELATIVE_PATH, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS, self.CLEAN_TARGET_FIELDNAME, self.INIT_DATA_RELATIVE_PATH, self.MODEL_BOOTSTRAP, self.MODEL_NUM_TREES, self.MODEL_MIN_SAMPLE_SPLIT, self.MODEL_MAX_FEATURES, self.MODEL_MAX_DEPTH, self.MODEL_CRITERION, self.MODEL_NUM_JOBS, self.MODEL_PERSIST, self.OPT_BEST_PARAMS_WRITE_PATH_UNDATED, self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID, self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID, self.OPT_MIN_SAMPLES_SPLIT_GRID, self.OPT_CRITERION_GRID, self.RFECV_MASK_WRITE_PATH_UNDATED, self.RFECV_MIN_FEATURES, self.RFECV_NUM_FOLDS, self.RFECV_SCORING, self.RFECV_STEP, self.TRAIN_METRICS_WRITE_PATH_UNDATED, self.TRAIN_SPLIT_PERCENTAGE, self.VALIDATION_PROBABLITY_THRESHOLD = self.CONFIGS.list_variables()
        self.CLEAN_MASK_PATH = self.get_absolute_path(self.CLEAN_MASK_RELATIVE_PATH)
        self.INIT_DATA_PATH = self.get_absolute_path(self.INIT_DATA_RELATIVE_PATH)
        self.RAW_DATA = pd.read_csv(self.INIT_DATA_PATH)
        self.RUN_DATE = dt.strftime(dt.now(), format='%Y_%m_%dT%H:%M:%s')

    def get_absolute_path(self, relative_path: str) -> str:
        """
        Provide absolute paths to files in user's environment.

        Args:
            relative_path: str | relative path of desired file inside of ENM-Protein-Predictor directory

        Returns: str | absolute path of desired file

        """
        return f"{self.PARENT_ABSOLUTE_PATH}/{relative_path}"

    def timestamp_filename(self, filename: str) -> str:
        """
        Add current run timestamp to filename to allow user to save multiple files and readily
        find the desired file.

        Args:
            filename: str | filename to add timestamp to

        Returns: str | filename adjusted by current timestamp

        """
        try:
            split_filename = filename.split('.')
        except IndexError as e:
            logging.error(f"Filename `{filename}` incorrectly formatted: {e}")
            raise
        return f"{split_filename[0]}-{self.RUN_DATE}.{split_filename[1]}"

    def clean_and_encode(self, df: pd.DataFrame) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        """
        Wrapper for data utils methods to multi-label encode, one-hot encode, and clean training data

        Args:
            df: pd.Dataframe | training data, including target

        Returns: Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]] | data post encoding
            and cleaning, split into features and target

        """
        d = data_utils.DataUtils()
        multi_label_encoded_train = d.multi_label_encode(df, self.CLEAN_MULTI_LABEL_ENCODE_FIELDS)
        one_hot_encoded_train = d.one_hot_encode(multi_label_encoded_train, self.CLEAN_CATEGORICAL_FIELDS)
        target = one_hot_encoded_train[self.CLEAN_TARGET_FIELDNAME]
        train_dropped_fields = one_hot_encoded_train.drop(self.CLEAN_DROP_FIELDS, axis=1)

        train_no_nulls = d.fill_nan_mean(train_dropped_fields, self.CLEAN_FILL_NAN_FIELDS)
        features = d.normalize(train_no_nulls)

        labels = d.classify(target, self.CLEAN_ENRICHMENT_SPLIT_VALUE)
        return features, labels

    def mask_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply binary mask to columns to remove features deemed less important by recursive
        feature elimination.

        Args:
            df: pd.Dataframe

        Returns: pd.Dataframe | features with masked columns dropped

        """
        d = data_utils.DataUtils()
        mask = d.get_mask(self.CLEAN_MASK_PATH)
        masked_features = d.apply_mask(mask, df)
        return masked_features

    def get_estimator(self) -> predictor_utils.RandomForestClassifierWithCoef:
        """
        Get Random Forest Classifier with parameters set from config file.

        Returns: predictor_utils.RandomForestClassifierWithCoef

        """
        est = predictor_utils.RandomForestClassifierWithCoef(
            n_estimators=self.MODEL_NUM_TREES,
            max_depth=self.MODEL_MAX_DEPTH,
            max_features=self.MODEL_MAX_FEATURES,
            criterion=self.MODEL_CRITERION,
            bootstrap=self.MODEL_BOOTSTRAP,
            min_samples_split=self.MODEL_MIN_SAMPLE_SPLIT,
            n_jobs=self.MODEL_NUM_JOBS
        )
        return est

    def train(self) -> Tuple[dict, dict, Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Clean and split data into train and validation, train a random forest classifier on
        training data, persist the trained model, and return metrics on validation data.

        Returns: Tuple[dict, dict, Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]] | validation metrics,
            information about feature importance, and classification information

        """
        d = data_utils.DataUtils()
        df = self.RAW_DATA
        train_accession_numbers = df[self.CLEAN_ACCESSION_NUMBERS_FIELDNAME]
        features, labels = self.clean_and_encode(df)
        features_masked = self.mask_features(features)
        train_features, train_labels, val_features, val_labels = d.split_data(self.TRAIN_SPLIT_PERCENTAGE, features_masked, labels)

        est = self.get_estimator()

        est.fit(train_features, train_labels)
        probability_prediction = pd.Series(est.predict_proba(val_features)[:, 1])

        val = validation_utils.ValidationUtils(val_labels, probability_prediction)
        classification_information = (probability_prediction, val_labels, train_accession_numbers, val_features)
        feature_importances = dict(zip(list(train_features), est.feature_importances_))
        validation_metrics = val.well_rounded_validation(self.VALIDATION_PROBABLITY_THRESHOLD)

        if self.MODEL_PERSIST['PERSIST']:
            filename = self.get_absolute_path(self.timestamp_filename(self.MODEL_PERSIST['PERSIST_PATH']))
            with open(filename, 'wb') as f:
                pickle.dump(est, f)

        validation_metrics_out_path = self.get_absolute_path(self.timestamp_filename(self.TRAIN_METRICS_WRITE_PATH_UNDATED))
        j = json.dumps(validation_metrics)
        with open(validation_metrics_out_path, 'w+') as f:
            f.write(j)

        return validation_metrics, feature_importances, classification_information

    def optimize(self) -> dict:
        """
        Clean raw data and run grid search on random forest classifier using grid defined in configs.

        Returns: dict | best parameters found via grid search

        """
        df = self.RAW_DATA
        features, labels = self.clean_and_encode(df)
        features_masked = self.mask_features(features)
        best_params_out_path = self.get_absolute_path(self.timestamp_filename(self.OPT_BEST_PARAMS_WRITE_PATH_UNDATED))
        est = self.get_estimator()
        optimized = est.optimize(self.OPT_NUM_FOLDS, self.OPT_NUM_TREES_GRID,
                     self.OPT_MAX_FEATURES_GRID, self.OPT_MAX_DEPTH_GRID,
                     self.OPT_MIN_SAMPLES_SPLIT_GRID, self.OPT_CRITERION_GRID,
                     features_masked, labels)
        best_params = optimized.best_params_
        print(f"Best parameters: \n {best_params}")
        j = json.dumps(best_params)
        with open(best_params_out_path, 'w+') as f:
            f.write(j)
        return best_params

    def rfecv(self) -> List[bool]:
        """
        Clean raw data and run recursive feature elimination to optimize training feature subset.

        Returns: List[bool] | mask to apply to features; True indicates that a features should be used
            in training, False that the feature should be dropped before training a model

        """
        df = self.RAW_DATA
        features, labels = self.clean_and_encode(df)
        est = self.get_estimator()
        mask_out_path = self.get_absolute_path(self.timestamp_filename(self.RFECV_MASK_WRITE_PATH_UNDATED))
        rfecv = est.recursive_feature_elimination(self.RFECV_STEP, self.RFECV_NUM_FOLDS, self.RFECV_MIN_FEATURES, self.RFECV_SCORING, features, labels)
        print(f"selector support: {rfecv.support_} \n selector ranking: {rfecv.ranking_}")
        print(f"Optimal number of features: {rfecv.n_features_} \n Selector grid scores: {rfecv.grid_scores_}")
        selector_support_list = rfecv.support_.tolist()
        with open(mask_out_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(selector_support_list)
        return selector_support_list
