import pandas as pd
from src.configs.constants import Constants
from typing import Literal


class DataHandler:
    def __init__(
        self,
        problem_type: Literal['regression', 'binary_class', 'multi_class'],
        data_source_filename: str,
        y_target_columns: str | list[str],
    ):
        self.filename = data_source_filename
        self.problem_type = problem_type
        self.dataframe = self.__create_dataframe()
        self.y_target_columns = self.__treat_y_target_columns(y_target_columns)
        self.columns = self.dataframe.columns.values.tolist()
        self.processed_data = self.get_processed_data()

    def __create_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(
            filepath_or_buffer=f'{Constants.DATA_DIRECTORY}/{self.__get_problem_type_data_path()}/{self.filename}',
            header=0,
            delimiter=Constants.CSV_DELIMITER,
        ).head(1) # TODO: REMOVE LATER!!!! JUST FOR TEMPORARY TESTING

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

    def __treat_binary_class_data(self) -> list[list[float]]:
        dataframe = self.dataframe.copy().drop(columns=self.y_target_columns)

        # Perform data manipulations

        return dataframe.values.tolist()

    def __treat_multi_class_data(self) -> list[list[float]]:
        dataframe = self.dataframe.copy().drop(columns=self.y_target_columns)

        # Perform data manipulations

        return dataframe.values.tolist()

    def __treat_regression_data(self) -> list[list[float]]:
        dataframe = self.dataframe.copy().drop(columns=self.y_target_columns)

        # Perform data manipulations

        return dataframe.values.tolist()

    def __treat_y_target_columns(self, y_target_columns: str | list[str]) -> list[str]:
        return y_target_columns if isinstance(y_target_columns, list) else [y_target_columns]

    def __get_y_target_columns_values(self):
        return {target_column: self.dataframe[target_column].values.tolist() for target_column in self.y_target_columns}

    def __process_problem_type_data(self) -> list[list[float]]:
        match self.problem_type:
            case 'regression':
                return self.__treat_regression_data()
            case 'binary_class':
                return self.__treat_binary_class_data()
            case 'multi_class':
                return self.__treat_multi_class_data()
            case _:
                raise Exception('Unknown problem type. DataHandler was unable to process problem_type dataframe.')

    def get_processed_data(self) -> tuple[list[list[float]], dict[str, list[float]]]:
        # Get normalized inputs and targets
        x: list[list[float]] = self.__process_problem_type_data()
        y_target: dict[str, list[float]] = self.__get_y_target_columns_values()

        return x, y_target
