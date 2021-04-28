import csv
import pandas as pd
import random
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
        indices = list(range(len(mask)))
        zipped = zip(indices, mask)
        indices_final = list(map(lambda x: x[0], list(filter(lambda x: x[1], list(zipped)))))
        masked_df = df[df.columns[indices_final]]
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
    def normalize_and_reshape(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        normalized_df = preprocessing.MinMaxScaler().fit_transform(df)
        normalized_with_columns = pd.DataFrame(normalized_df, columns=list(df))
        labelled = pd.concat([labels, normalized_with_columns], axis=1)
        labelled_reset = labelled.reset_index(drop=True)
        return labelled_reset

    @staticmethod
    def save_accession_numbers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        accession_numbers = df['Accesion Number']
        return df.drop('Accesion Number', axis=1), accession_numbers

    @staticmethod
    def split_data(train_percent: float, train: pd.DataFrame, target: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_features, val_features, train_target, val_target = model_selection.train_test_split(
            train,
            target,
            test_size=train_percent,
            stratify=target,
            random_state=int((random.random() * 100)))
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

    def clean_raw_data(self, df: pd.DataFrame, drop_fields: List[str], fill_nan_fields: List[str], enrichment_split_value: float) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        """ Cleans the raw data, drops useless columns, one hot encodes, and extracts
        class information

        Args, Returns: None
        """

        enrichment = df['Enrichment']
        accesion_numbers = df['Accesion Number']
        one_hot_encoded_train_dropped_fields = df.drop(drop_fields, axis=1)

        d = DataUtils()
        train_no_nulls = d.fill_nan_mean(one_hot_encoded_train_dropped_fields, fill_nan_fields)
        cleaned_train = d.normalize_and_reshape(train_no_nulls, accesion_numbers)

        target = d.classify(enrichment, enrichment_split_value)
        return cleaned_train, target






def clean_print(obj):
    """
    Prints the JSON in a clean format for all my
    Biochemistry friends

    Args:
        :param obj (object): Any object you wish to print in readable format

    Returns:
        None
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            if hasattr(val, '__iter__'):
                print(f"{key}")
                clean_print(val)
            else:
                print(f"{key}: {val}")
    elif isinstance(obj, list):
        for val in obj:
            if hasattr(val, '__iter__'):
                clean_print(val)
            else:
                print(f"{val}")
    else:
        if isinstance(obj, pd.DataFrame):
            clean_print(obj.to_dict(orient='records'))
        else:
            print(f"{str(obj)}")


def to_excel(classification_information):
    """ Prints model output to an excel file

        Args:
            :classification_information (numpy array): Information about results
            >classification_information = {
                'all_predict_proba' : np.empty([TOTAL_TESTED_PROTEINS], dtype=float),
                'all_true_results' : np.empty([TOTAL_TESTED_PROTEINS], dtype=int),
                'all_accesion_numbers' : np.empty([TOTAL_TESTED_PROTEINS], dtype=str),
                'all_particle_information' : np.empty([2, TOTAL_TESTED_PROTEINS], dtype=int),
                'all_solvent_information' : np.empty([3, TOTAL_TESTED_PROTEINS], dtype=int)
                }
        Returns:
            None
        """
    with open('prediction_probability.csv', 'w') as file:
        file.write('Protein Accesion Number, Particle Type, Solvent Conditions, True Bound Value, Predicted Bound Value, Predicted Probability of Being Bound, Properly Classified\n')

        for pred, true_val, protein, particle_s, particle_c, cys, salt8, salt3, in zip(classification_information['all_predict_proba'],
                                                                                       classification_information['all_true_results'],
                                                                                       classification_information['all_accesion_numbers'],
                                                                                       classification_information['all_particle_information'][0],
                                                                                       classification_information['all_particle_information'][1],
                                                                                       classification_information['all_solvent_information'][0],
                                                                                       classification_information['all_solvent_information'][1],
                                                                                       classification_information['all_solvent_information'][2]
                                                                                       ):
            bound = 'no'
            predicted_bound = 'no'
            properly_classified = 'no'
            particle_charge = 'negative'
            particle_size = '10nm'
            solvent = '10 mM NaPi pH 7.4'

            if int(round(pred)) == true_val:
                properly_classified = 'yes'
            if true_val == 1:
                bound = 'yes'
            if int(round(pred)) == 1:
                predicted_bound = 'yes'
            if particle_s == 0:
                particle_size = '100nm'
            if particle_c == 1:
                particle_charge = 'positive'
            if (particle_size == '10nm' and particle_charge == 'positive'):
                particle = '(+) 10 nm AgNP'
            if (particle_size == '10nm' and particle_charge == 'negative'):
                particle = '(-) 10 nm AgNP'
            if (particle_size == '100nm' and particle_charge == 'negative'):
                particle = '(-) 100 nm AgNP'
            if (cys == 1):
                solvent = '10 mM NaPi pH 7.4 + 0.1 mM cys'
            if (salt8 == 1):
                solvent = '10 mM NaPi pH 7.4 + 0.8 mM NaCl'
            if (salt3 == 1):
                solvent = '10 mM NaPi pH 7.4 + 3.0 mM NaCl'

            file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(protein, particle, solvent, bound, predicted_bound,round(pred, 2), properly_classified))


def hold_in_memory(classification_information, metrics, iterations, test_size):
    """Holds classification data in memory to be exported to excel

    Args:
        :classification_information (dict): container for all the classification_information from all the runs
        :metrics (tuple): information from the current test set to add to classification_information
        :iterations (int): The current test iterations
        :test_size (int): The amount of values in the current test set
    Returns:
        None
    """
    i = iterations
    TEST_SIZE = test_size #10% of training data is used for testing ceil(10% of 3012)=302
    PARTICLE_SIZE = 0
    PARTICLE_CHARGE = 1
    SOLVENT_CYS = 0
    SOLVENT_SALT_08 = 1
    SOLVENT_SALT_3 = 2
    #Information is placed into numpy arrays as blocks
    classification_information['all_predict_proba'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[0]
    classification_information['all_true_results'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[1]
    classification_information['all_accesion_numbers'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[2]
    classification_information['all_particle_information'][PARTICLE_CHARGE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Particle Charge_1']
    classification_information['all_particle_information'][PARTICLE_SIZE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Particle Size_10']
    classification_information['all_solvent_information'][SOLVENT_CYS][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent Cysteine Concentration_0.1']
    classification_information['all_solvent_information'][SOLVENT_SALT_08][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_0.8']
    classification_information['all_solvent_information'][SOLVENT_SALT_3][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_3.0']
