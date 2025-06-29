import math

import pandas as pd
from src.configs.constants import Constants
from typing import Literal


class DataHandler:
    def __init__(
            self,
            problem_type: Literal['regression', 'binary_class', 'multi_class'],
            dataset_name: Literal['mushroom', 'students', 'bike'],
            y_target_columns: str | list[str],
            dataset_type: Literal['complete', 'training', 'validation', 'test'],
    ):
        self.problem_type = problem_type

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.dataset_filename = self.__get_filename_by_dataset_type()
        self.dataframe = self.__create_dataframe(self.dataset_filename)

        self.y_target_columns = self.__treat_y_target_columns(y_target_columns)
        self.columns = self.dataframe.columns.values.tolist()

        self.processed_data = self.__get_processed_data(self.dataframe)

    def __get_filename_by_dataset_type(self):
        slice_information = f'_{self.dataset_type}_slice' if self.dataset_type != 'complete' else ''
        return f'{self.dataset_name}_converted{slice_information}.csv'

    def __create_dataframe(self, csv_file: str) -> pd.DataFrame:
        slice_path = 'slices/' if self.dataset_type != 'complete' else ''
        return pd.read_csv(
            filepath_or_buffer=f'{Constants.DATA_DIRECTORY}/{self.__get_problem_type_data_path()}/{slice_path}{csv_file}',
            header=0,
            delimiter=Constants.CSV_DELIMITER,
        )

    def __get_problem_type_data_path(self) -> str:
        match self.problem_type:
            case 'regression':
                return Constants.REGRESSION_DATA_FILEPATH
            case 'binary_class':
                return Constants.BINARY_CLASS_DATA_FILEPATH
            case 'multi_class':
                return Constants.MULTI_CLASS_DATA_FILEPATH
            case _:
                raise Exception('Unknown problem type. DataHandler was unable to read file data.')

    def __treat_y_target_columns(self, y_target_columns: str | list[str]) -> list[str]:
        return y_target_columns if isinstance(y_target_columns, list) else [y_target_columns]

    def get_processed_data_slices(
            self,
            test_set_percentage: float,
            validation_set_percentage: float,
            persist_to_csv: bool = False,
    ) -> tuple[
        tuple[list[list[float]], dict[str, list[float]]],  # validation
        tuple[list[list[float]], dict[str, list[float]]],  # train
        tuple[list[list[float]], dict[str, list[float]]]  # test
    ]:
        if not all([self.dataset_type == 'complete', 'converted' in self.dataset_filename]):
            raise Exception('Dataset type and must be "complete" to create slices.')

        shuffled_dataframe = self.dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
        test_data_rows = math.ceil(shuffled_dataframe.shape[0] * test_set_percentage)

        test_dataframe = self.dataframe.iloc[:test_data_rows]
        # Temporary Training data slice without extracting Validation
        temp_training = self.dataframe.iloc[test_data_rows:]

        # Get Validation and Training Dataframe slices
        validation_rows = math.ceil(temp_training.shape[0] * validation_set_percentage)
        training_dataframe = temp_training.iloc[validation_rows:]
        validation_dataframe = temp_training.iloc[:validation_rows]

        # Persist csv:
        if persist_to_csv:
            original_csv_filename = self.dataset_filename.split('.')[0]
            common_path = f'{Constants.DATA_DIRECTORY}/{self.__get_problem_type_data_path()}/slices'
            validation_dataframe.to_csv(f'{common_path}/{original_csv_filename}_validation_slice.csv', index=False)
            training_dataframe.to_csv(f'{common_path}/{original_csv_filename}_training_slice.csv', index=False)
            test_dataframe.to_csv(f'{common_path}/{original_csv_filename}_test_slice.csv', index=False)

        # Process Dataframes slices
        validation_processed_data = self.__get_processed_data(validation_dataframe)
        training_processed_data = self.__get_processed_data(training_dataframe)
        test_processed_data = self.__get_processed_data(test_dataframe)

        return validation_processed_data, training_processed_data, test_processed_data

    def __get_processed_data(self, df: pd.DataFrame) -> tuple[list[list[float]], dict[str, list[float]]]:
        # Get normalized inputs and targets
        x: list[list[float]] = self.__process_problem_type_data(df)
        y_target: dict[str, list[float]] = self.__get_y_target_columns_values(df)

        return x, y_target

    def __process_problem_type_data(self, df: pd.DataFrame) -> list[list[float]]:
        match self.problem_type:
            case 'regression':
                return self.__treat_regression_data(df)
            case 'binary_class':
                return self.__treat_binary_class_data(df)
            case 'multi_class':
                return self.__treat_multi_class_data(df)
            case _:
                raise Exception('Unknown problem type. DataHandler was unable to process problem_type dataframe.')

    def __treat_regression_data(self, df: pd.DataFrame) -> list[list[float]]:
        dataframe = df.copy().drop(columns=self.y_target_columns)

        # Perform data manipulations

        return dataframe.values.tolist()

    def __treat_binary_class_data(self, df: pd.DataFrame) -> list[list[float]]:
        dataframe = df.copy().drop(columns=self.y_target_columns)

        # Perform data manipulations

        return dataframe.values.tolist()

    def __treat_multi_class_data(self, df: pd.DataFrame) -> list[list[float]]:
        dataframe = df.copy().drop(columns=self.y_target_columns)

        # Perform data manipulations

        return dataframe.values.tolist()

    def __get_y_target_columns_values(self, df: pd.DataFrame = None):
        return {target_column: df[target_column].values.tolist() for target_column in self.y_target_columns}
