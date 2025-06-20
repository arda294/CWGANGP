from rdt.transformers.numerical import ClusterBasedNormalizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from joblib import Parallel, delayed
import os
import numpy as np

class DataColumnInfo:
    def __init__(self, is_discrete, column_name, category_counts = None, gmm = None):
        self.is_discrete = is_discrete
        self.column_name = column_name
        self.category_counts = category_counts
        self.gmm = gmm
        self.num_clusters = 0

    def __repr__(self):
        return f"column_name={self.column_name}, gmm={self.gmm}, num_clusters={self.num_clusters}, is_discrete={self.is_discrete}"
    
class DataTransformer:
    """
    Parameters
    ----------
    fit_n_jobs : int, default=0
        Number of parallel jobs for fitting data. 
        If 0, uses all available CPU cores (Not Recommended).
        
    transform_n_jobs : int, default=0
        Number of parallel jobs for transforming data.
        If 0, uses all available CPU cores.
    """
    def __init__(self, fit_n_jobs=0, transform_n_jobs=0):
        self.fit_n_jobs = fit_n_jobs
        self.transform_n_jobs = transform_n_jobs
        self.discrete_transformer = OneHotEncoder(sparse_output=False)
        self.data_column_infos = {}
        self.raw_data_column_order = []

    @staticmethod
    def load_state(fit_n_jobs, transform_n_jobs, discrete_transformer, data_column_infos, raw_data_column_order):
        transformer = DataTransformer(fit_n_jobs=fit_n_jobs, transform_n_jobs=transform_n_jobs)

        transformer.discrete_transformer = discrete_transformer
        transformer.data_column_infos = data_column_infos
        transformer.raw_data_column_order = raw_data_column_order

        return transformer

    def fit(self, raw_data: pd.DataFrame, discrete_columns: list) -> None:
        """
        Fits raw data to the data transformer, clears previous column data infos
        """

        # Clear previus column info
        self.data_column_infos = {}
        self.raw_data_column_order = raw_data.columns.tolist()

        self._fit_discrete(raw_data, discrete_columns)
        self._fit_numerical(raw_data, discrete_columns)

    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data
        """
        
        cond_cols = self._transform_discrete(raw_data)
        numerical = self._transform_numerical(raw_data)

        # Prepare one hot encoder for converting label encoded mode indicators
        enc = OneHotEncoder(sparse_output=False)
        numerical_cols = []

        for col, col_name in numerical:
            # Convert to one hot encoding
            mode_onehot = enc.fit_transform(col[[f"{col_name}.component"]])
            num_clusters = mode_onehot.shape[1]
            # Add new columns
            for i in range(num_clusters):
                col[f"{col_name}.{i}"] = mode_onehot[:,i]
            col = col.drop([f"{col_name}.component"], axis=1)
            self.data_column_infos[col_name].num_clusters = num_clusters
            numerical_cols.append(col)

        conditional_df = pd.DataFrame(cond_cols)
        if len(numerical_cols) == 0:
            return conditional_df
        numerical_df = pd.concat([*numerical_cols], axis=1, join='inner')
        return pd.concat([numerical_df, conditional_df], axis=1, join='inner')

    def inverse_transform(self, transformed_data: pd.DataFrame, parallel: bool = False) -> pd.DataFrame:
        """
        Inverse transform already transformed data
        """
        inverse_cond, inverse_numerical = self._inverse_transform_discrete(transformed_data), self._inverse_transform_numerical(transformed_data, parallel=parallel)
        cond_df = pd.DataFrame(inverse_cond, columns=self.discrete_transformer.feature_names_in_)
        if len(inverse_numerical) == 0:
            return cond_df[self.raw_data_column_order]     
        numerical_df = pd.concat([*inverse_numerical], axis=1, join='inner')
        return pd.concat([cond_df, numerical_df], axis=1, join='inner')[self.raw_data_column_order]     

    def category_counts(self) -> list[dict]:
        # Get category counts
        category_counts = []
        for col_info in [col_info for _, col_info in self.data_column_infos.items() if col_info.is_discrete == True]:
            category_counts.append(col_info.category_counts)
        return category_counts

    def _fit_discrete(self, raw_data: pd.DataFrame, discrete_columns: list) -> None:
        """
        Fits discrete raw data
        """
        # Get valid discrete columns
        columns = [col for col in raw_data.columns if col in discrete_columns]
        
        self.discrete_transformer.fit(raw_data[[*columns]])

        for col in columns:
            self.data_column_infos[col] = DataColumnInfo(
                is_discrete=True,
                category_counts=raw_data.value_counts(col, sort=False).to_dict(),
                gmm=None,
                column_name=col
            )

    def _fit_numerical(self, raw_data: pd.DataFrame, discrete_columns: list) -> None:
        """
        Fit numerical columns in parallel
        """
        def _fit_helper(column_name):
            gmm = ClusterBasedNormalizer(missing_value_generation='from_column')
            gmm.fit(raw_data[[column_name]], column_name)

            return gmm, column_name
        
        # Get numerical columns
        numerical_columns = [col for col in raw_data.columns if col not in discrete_columns]

        # Prepare parallel delayed calls
        delayed_calls = []
        for column_name in numerical_columns:
            delayed_call = delayed(_fit_helper)(column_name)
            delayed_calls.append(delayed_call)

        num_jobs = self.fit_n_jobs if self.fit_n_jobs else os.cpu_count()
        print(f"fit numerical with {num_jobs} jobs...")

        # Run delayed calls
        try:
            results = Parallel(n_jobs=num_jobs, verbose=10)(delayed_calls)
            for gmm, column_name in results:
                self.data_column_infos[column_name] = DataColumnInfo(
                    is_discrete=False,
                    gmm=gmm,
                    column_name=column_name
                )

        except KeyboardInterrupt:
            print('Stopping...')
            raise

    def _transform_discrete(self, raw_data: pd.DataFrame):
        """
        Transforms discrete raw data
        """
        # Get discrete columns
        discrete_columns = [col_name for col_name, col_info in self.data_column_infos.items() if col_info.is_discrete]
        return self.discrete_transformer.transform(raw_data[[*discrete_columns]])

    def _transform_numerical(self, raw_data: pd.DataFrame):
        """
        Transforms numerical columns in parallel
        """
        def _transform_helper(col_info: DataColumnInfo):
            transformed_col = col_info.gmm.transform(raw_data[[col_info.column_name]])
            return transformed_col, col_info.column_name
        
        # Prepare parallel delayed calls
        delayed_calls = []
        for _, col_info in self.data_column_infos.items():
            if not col_info.is_discrete:
                delayed_call = delayed(_transform_helper)(col_info)
                delayed_calls.append(delayed_call)

        num_jobs = self.transform_n_jobs if self.transform_n_jobs else os.cpu_count()
        print(f"transform numerical with {num_jobs} jobs...")

        # Run delayed calls
        try:
            results = Parallel(n_jobs=num_jobs, verbose=10)(delayed_calls)
        except KeyboardInterrupt:
            print('Stopping...')
            raise

        return results

    def _inverse_transform_discrete(self, transformed_data: pd.DataFrame):
        """
        Inverse transform discrete data
        """
        one_hot_dim = 0

        for column_categories in self.discrete_transformer.categories_:
            one_hot_dim += len(column_categories)

        return self.discrete_transformer.inverse_transform(transformed_data.iloc[:,-one_hot_dim:])

    def _inverse_transform_numerical(self, transformed_data: pd.DataFrame, parallel: bool = False):
        """
        Inverse transform numerical data
        """

        transformed_data = transformed_data.copy()

        def _transform_helper(col_info: DataColumnInfo):
            # Get one hot mode indicators
            onehot_cols = [f"{col_info.column_name}.{idx}" for idx in range(col_info.num_clusters)]
            mode_onehot = transformed_data[onehot_cols]
            # Convert to label encoding
            transformed_data[f"{col_info.column_name}.component"] = np.argmax(mode_onehot, axis=1)
            return col_info.gmm.reverse_transform(transformed_data[[col_info.column_name + ".normalized", col_info.column_name + ".component"]])
        
        if parallel:
            return self._inverse_transform_numerical_parallel(transform_func=_transform_helper)
        else:
            return self._inverse_transform_numerical_sequential(transform_func=_transform_helper)

    def _inverse_transform_numerical_parallel(self, transform_func: callable):
        # Prepare parallel delayed calls
        delayed_calls = []
        for _, col_info in self.data_column_infos.items():
            if not col_info.is_discrete:
                delayed_call = delayed(transform_func)(col_info)
                delayed_calls.append(delayed_call)

        num_jobs = self.transform_n_jobs if self.transform_n_jobs else os.cpu_count()
        print(f"inverse transform numerical with {num_jobs} jobs...")

        # Run delayed calls
        try:
            results = Parallel(n_jobs=num_jobs, verbose=10)(delayed_calls)
        except KeyboardInterrupt:
            print('Stopping...')
            raise

        return results
    
    def _inverse_transform_numerical_sequential(self, transform_func: callable):
        results = []
        for _, col_info in self.data_column_infos.items():
            if not col_info.is_discrete:
                results.append(transform_func(col_info))

        return results
