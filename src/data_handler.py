import pandas as pd
from src.configs.constants import Constants
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataHandler:
    def __init__(
            self,
            dataset_type: Literal['complete', 'training', 'validation', 'test'],
            problem_type: Literal['regression', 'binary_class', 'multi_class'],
            dataset_name: Literal['mushroom', 'students', 'bike'],
            y_target_columns: str | list[str],
    ):
        self.dataset_type = dataset_type
        self.problem_type = problem_type
        self.dataset_name = dataset_name
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

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

        temp_df, test_dataframe = train_test_split(self.dataframe, test_size=test_set_percentage, random_state=42)
        training_dataframe, validation_dataframe = train_test_split(temp_df, test_size=validation_set_percentage, random_state=42)

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
        x: list[list[float]] = self.__get_x_processed_data(df)
        y_target: dict[str, list[float]] = self.__get_y_targets_processed_data(df)

        return x, y_target

    def __get_x_processed_data(self, df: pd.DataFrame) -> list[list[float]]:
        dataframe = df.copy().drop(columns=self.y_target_columns)

        dataframe_scaled = self.x_scaler.fit_transform(dataframe)

        return dataframe_scaled.tolist()

    def __get_y_targets_processed_data(self, df: pd.DataFrame = None):
        return {target_column: self.y_scaler.fit_transform(df[target_column].values.reshape(-1, 1)).flatten().tolist() if self.problem_type == 'regression' else df[target_column].values.tolist() for target_column in self.y_target_columns}
