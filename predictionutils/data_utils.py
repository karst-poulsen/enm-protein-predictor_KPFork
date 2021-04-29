import csv
import pandas as pd
from sklearn import preprocessing, model_selection
from typing import List, Tuple, Union


class DataUtils:

    def __init__(self):
        self.init = True

    @staticmethod
    def get_mask(path: str) -> List[bool]:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            column_mask = list(map(lambda x: x == "True", list(reader)[0]))
        return column_mask

    @staticmethod
    def apply_mask(mask: List[bool], df: pd.DataFrame) -> pd.DataFrame:
        column_indices = list(range(len(mask)))
        zipped = zip(column_indices, mask)
        column_indices_final = list(map(lambda x: x[0], list(filter(lambda x: x[1], list(zipped)))))
        masked_df = df[df.columns[column_indices_final]]
        return masked_df

    def classify(self, col: Union[pd.DataFrame, pd.Series], cutoff: float) -> Union[pd.DataFrame, pd.Series]:
        return col.apply(lambda x: self.classify_value(x, cutoff))

    @staticmethod
    def classify_value(val: float, cutoff: float) -> int:
        if val >= cutoff:
            classification = 1
        else:
            classification = 0
        return classification

    def fill_nan_mean(self, df: pd.DataFrame, field_names: List[str]) -> pd.DataFrame:
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
    def normalize_and_reshape(df: pd.DataFrame) -> pd.DataFrame:
        normalized_df = preprocessing.MinMaxScaler().fit_transform(df)
        normalized_with_columns = pd.DataFrame(normalized_df, columns=list(df))
        normalized_reset = normalized_with_columns.reset_index(drop=True)
        return normalized_reset

    @staticmethod
    def save_accession_numbers(df: pd.DataFrame, accession_number_fieldname: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        accession_numbers = df[accession_number_fieldname]
        return accession_numbers

    @staticmethod
    def split_data(train_percent: float, train: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_features, val_features, train_target, val_target = model_selection.train_test_split(
            train,
            target,
            train_size=train_percent,
            stratify=target)
        return train_features, train_target, val_features, val_target

    def one_hot_encode(self, df: pd.DataFrame, field_names: List[str]) -> pd.DataFrame:
        if len(field_names) == 0:
            return df
        else:
            field_name = field_names[0]
            dummy = pd.get_dummies(df[field_name], prefix=field_name)
            df_with_dummies = pd.concat([df, dummy], axis=1)
            df_dropped_cols = df_with_dummies.drop(field_name, axis=1)
            return self.one_hot_encode(df_dropped_cols, field_names[1:])

    def multi_label_encode(self, df: pd.DataFrame, field_names: List[str]) -> pd.DataFrame:
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


class Database(object):
    """Handles all data fetching and preparation. Attributes
       can be assigned to csv files with the assignment operator. Typical use
       case is to set raw_data to a csv file matching the format found in
       Input files and then calling clean_raw_data(). This sets the clean_X_data,
       y_enrichment and target values. From this point you can split the data
       to train/test the model using our data. To predict your own data, make sure your excel sheet
       matches the format in <Input_Files/database.csv>. Then you can
       call db.predict = <your_csv_path>. The X_test and Y_test data will now
       be your data. Just remove the stratified_data_split from the pipeline
       because you will now not need to split any data.

       Args:
            None
       Attributes:
            :self._raw_data (Pandas Dataframe): Holds raw data in the same form as excel file. initialized after fetch_raw_data() is called
            :self._clean_X_data (Pandas Dataframe): Holds cleaned and prepared X data.
            :self._Y_enrichment (numpy array): Holds continous Y values
            :self._X_train (Pandas Dataframe): Holds the X training data
            :self._X_test (Pandas Dataframe): Holds the X test data
            :self._Y_train (Pandas Dataframe): Holds the Y training data
            :self._Y_test (Pandas Dataframe): Holds the T testing data
            :self._test_accesion_numbers (list): holds the accesion_numbers
            in the test set
        """

    def __init__(self, raw_data_path: str):
        self.DATA_PATH = raw_data_path
        self.RAW_DATA = pd.read_csv(self.DATA_PATH)

    def clean_raw_data(self, df: pd.DataFrame, drop_fields: List[str], fill_nan_fields: List[str], enrichment_split_value: float, target_fieldname: str) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        """ Cleans the raw data, drops useless columns, one hot encodes, and extracts
        class information

        Args, Returns: None
        """

        target = df[target_fieldname]
        df_dropped_fields = df.drop(drop_fields, axis=1)

        d = DataUtils()
        df_no_nulls = d.fill_nan_mean(df_dropped_fields, fill_nan_fields)
        cleaned_train = d.normalize_and_reshape(df_no_nulls)

        target_classified = d.classify(target, enrichment_split_value)
        return cleaned_train, target_classified
