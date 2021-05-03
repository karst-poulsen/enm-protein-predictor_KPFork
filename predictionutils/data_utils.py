import csv
import pandas as pd
from sklearn import preprocessing, model_selection
from typing import List, Tuple, Union


class DataUtils:
    """
    Utility class to hold common cleaning methods.
    """
    def __init__(self):
        self.init = True

    @staticmethod
    def get_mask(path: str) -> List[bool]:
        """
        Read binary mask into memory.

        Args:
            path: str | path to mask file

        Returns: List[bool] | binary mask to apply to columns

        """
        with open(path, 'r') as f:
            reader = csv.reader(f)
            column_mask = list(map(lambda x: x == "True", list(reader)[0]))
        return column_mask

    @staticmethod
    def apply_mask(mask: List[bool], df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply binary mask to drop columns ranked as less important by recursive feature elimination.

        Args:
            mask: List[bool] | binary mask -- True if we should keep the column and False otherwise
            df: pd.DataFrame

        Returns: pd.DataFrame | original dataframe without columns defined by mask

        """
        column_indices = list(range(len(mask)))
        zipped = zip(column_indices, mask)
        column_indices_final = list(map(lambda x: x[0], list(filter(lambda x: x[1], list(zipped)))))
        masked_df = df[df.columns[column_indices_final]]
        return masked_df

    def classify(self, col: Union[pd.DataFrame, pd.Series], cutoff: float) -> Union[pd.DataFrame, pd.Series]:
        """
        Apply value classification function to dataframe.

        Args:
            col: Union[pd.DataFrame, pd.Series] | Target data
            cutoff: float | cutoff value for classification

        Returns: Union[pd.DataFrame, pd.Series] | classified data

        """
        return col.apply(lambda x: self.classify_value(x, cutoff))

    @staticmethod
    def classify_value(val: float, cutoff: float) -> int:
        """
        Function to classify target into binary labels.

        Args:
            val: float | value to classify
            cutoff: float | cutoff value for classification

        Returns: 0 if val < cutoff ese 1

        """
        if val >= cutoff:
            classification = 1
        else:
            classification = 0
        return classification

    def fill_nan_mean(self, df: pd.DataFrame, field_names: List[str]) -> pd.DataFrame:
        """
        Fill nan values with the column's mean value.
        Args:
            df: pd.DataFrame
            field_names: List[str] | List of fieldnames to fill

        Returns: pd.Dataframe

        """
        if len(field_names) == 0:
            return df
        else:
            field_name = field_names[0]
            s = df.sum(axis=0)[field_name]
            c = df[field_name].index.size
            mean = s / c
            updated_field_name = f"{field_name}_updated"
            df[updated_field_name] = df[field_name].fillna(mean)
            df_no_nulls = df.drop(field_name, axis=1)
            df_correct_field_names = df_no_nulls.rename({updated_field_name: field_name}, axis=1)
            return self.fill_nan_mean(df_correct_field_names, field_names[1:])

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize columns.

        Args:
            df: pd.DataFrame

        Returns: pd.DataFrame | dataframe with continuous columns normalized via scikit learn's MinMaxScaler

        """
        normalized_df = preprocessing.MinMaxScaler().fit_transform(df)
        normalized_with_columns = pd.DataFrame(normalized_df, columns=list(df))
        normalized_reset = normalized_with_columns.reset_index(drop=True)
        return normalized_reset


    @staticmethod
    def split_data(train_percent: float, train: pd.DataFrame, labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.

        Args:
            train_percent: float | what percent of data to use for training
            train: pd.DataFrame | features dataframe
            labels: pd.DataFrame | labels dataframe

        Returns: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | training features,
        training labels, validation features, validation labels

        """
        train_features, val_features, train_labels, val_labels = model_selection.train_test_split(
            train,
            labels,
            train_size=train_percent,
            stratify=labels)
        return train_features, train_labels, val_features, val_labels

    def one_hot_encode(self, df: pd.DataFrame, field_names: List[str]) -> pd.DataFrame:
        """
        One-hot encode all fields in `field_names` and attach as columns to original dataframe.
        Args:
            df: pd.Dataframe
            field_names: List[str] | fieldnames of columns to one-hot encode

        Returns: original df with new one-hot encoded columns minus old unencoded columns

        """
        if len(field_names) == 0:
            return df
        else:
            field_name = field_names[0]
            dummy = pd.get_dummies(df[field_name], prefix=field_name)
            df_with_dummies = pd.concat([df, dummy], axis=1)
            df_dropped_cols = df_with_dummies.drop(field_name, axis=1)
            return self.one_hot_encode(df_dropped_cols, field_names[1:])

    def multi_label_encode(self, df: pd.DataFrame, field_names: List[str]) -> pd.DataFrame:
        """
        Multi-label encode all fields in `field_names` and attach as columns to original dataframe.

        Args:
            df: pd.Dataframe
            field_names: List[str] | fieldnames of columns to multi-label encode

        Returns: original df with new multi-label encoded columns minus old unencoded columns

        """
        if len(field_names) == 0:
            return df
        else:
            df_reset = df.reset_index(drop=True)
            field_name = field_names[0]
            split_field_name = f"{field_name}_split"
            df_reset[split_field_name] = df_reset[field_name].apply(lambda x: x.split(';'))
            multi_dim_list = list(map(lambda x: x.split(';'), df_reset[field_name].values))
            flattened_list = [x for lst in multi_dim_list for x in lst]
            val_set = list(filter(lambda x: x != '', set(flattened_list)))
            for val in val_set:
                df_reset[val] = df_reset[split_field_name].apply(lambda x: 1 if val in x else 0)
            cleaned_df = df_reset.drop([field_name, split_field_name, '0'], axis=1)
            return self.multi_label_encode(cleaned_df, field_names[1:])
