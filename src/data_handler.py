import math

import pandas as pd
from src.configs.constants import Constants
from typing import Literal


class DataHandler:
    def __init__(
            self,
            problem_type: Literal['regression', 'binary_class', 'multi_class'],
            data_source_filename: str,
            y_target_columns: str | list[str],
            auto_process: bool = False,
            persist_to_csv: bool = False,
    ):
        self.filename = data_source_filename
        self.problem_type = problem_type
        self.dataframe = self.__create_dataframe()
        self.y_target_columns = self.__treat_y_target_columns(y_target_columns)
        self.columns = self.dataframe.columns.values.tolist()

        self.__total_processed_data = self.__get_processed_data(self.dataframe)
        self.test_data_percentage: float = 0.2
        self.validation_data_percentage: float = 0.2
        self.validation_data = None
        self.training_data = None
        self.test_data = None
        if auto_process:
            self.get_processed_data_slices(persist_to_csv=persist_to_csv)

    def __create_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=f'{Constants.DATA_DIRECTORY}/{self.__get_problem_type_data_path()}/{self.filename}',
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

    def get_processed_data_slices(self, persist_to_csv: bool = False) -> tuple[
        tuple[list[list[float]], dict[str, list[float]]],  # validation
        tuple[list[list[float]], dict[str, list[float]]],  # train
        tuple[list[list[float]], dict[str, list[float]]]  # test
    ]:
        shuffled_dataframe = self.dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
        test_data_rows = math.ceil(shuffled_dataframe.shape[0] * self.test_data_percentage)

        test_dataframe = self.dataframe.iloc[:test_data_rows]
        # Temporary Training data slice without extracting Validation
        temp_training = self.dataframe.iloc[test_data_rows:]

        # Get Validation and Training Dataframe slices
        validation_rows = math.ceil(temp_training.shape[0] * self.validation_data_percentage)
        training_dataframe = temp_training.iloc[validation_rows:]
        validation_dataframe = temp_training.iloc[:validation_rows]

        # Persist csv:
        if persist_to_csv:
            original_csv_filename = self.filename.split('.')[0]
            common_path = f'{Constants.DATA_DIRECTORY}/{self.__get_problem_type_data_path()}/slices'
            validation_dataframe.to_csv(f'{common_path}/{original_csv_filename}_validation_slice.csv', index=False)
            training_dataframe.to_csv(f'{common_path}/{original_csv_filename}_training_slice.csv',index=False)
            test_dataframe.to_csv(f'{common_path}/{original_csv_filename}_test_slice.csv',index=False)

        # Process Dataframes slices
        self.validation_data = self.__get_processed_data(validation_dataframe)
        self.training_data = self.__get_processed_data(training_dataframe)
        self.test_data = self.__get_processed_data(test_dataframe)

        return self.validation_data, self.training_data, self.test_data

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
