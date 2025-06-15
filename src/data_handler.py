import pandas as pd
from src.configs.constants import Constants
from typing import Literal


class DataHandler:
    def __init__(
            self,
            problem_type: Literal['regression', 'binary_class', 'multi_class'],
            data_source_filename: str,
            y_target_column_name: str,
    ):
        self.filename = data_source_filename
        self.problem_type = problem_type
        self.dataframe = self.create_dataframe()
        self.y_target_column_name = y_target_column_name
        self.columns = self.dataframe.columns.values.tolist()
        self.processed_data = self.process_problem_type_data()

    def create_dataframe(self) -> pd.DataFrame:
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

    def treat_binary_class_data(self) -> pd.DataFrame:
        ...

    def treat_multi_class_data(self) -> pd.DataFrame:
        ...

    def treat_regression_data(self) -> pd.DataFrame:
        ...

    def process_problem_type_data(self) -> tuple[list[list[float]], list[float]]:
        match self.problem_type:
            case 'regression':
                self.dataframe = self.treat_regression_data()
            case 'binary_class':
                self.dataframe = self.treat_binary_class_data()
            case 'multi_class':
                self.dataframe = self.treat_multi_class_data()
            case _:
                raise Exception('Unknown problem type. DataHandler was unable to process problem_type dataframe.')

        return self.__convert_df_to_lists()

    def __convert_df_to_lists(self) -> tuple[list[list[float]], list[float]]:
        # Perform data manipulations
        x: list[list[float]] = [[]]
        y_target: list[float] = []

        return x, y_target
