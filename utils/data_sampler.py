import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import pandas as pd


class TabularDataset(Dataset):
    """
    Parameters
    ----------
    dataset : str or pd.DataFrame
        Path to CSV file or pandas DataFrame containing the data.
    discrete_column_counts : list[dict], optional
        List containing dictionaries that define the discrete columns.
        Each dictionary maps category values to their counts in the dataset.
        Used for tracking discrete column distributions.
    header_row : bool, default=False
        Whether the CSV file contains a header row that should be skipped.
    label_column : int, default=-1
        Index of the column to use as target/label (y). By default, uses the last column.
        All other columns are used as features (x).
    dtype : numpy.dtype, default=np.float32
        Data type to use for the dataset.
    """
    def __init__(self, dataset: (str | pd.DataFrame), numerical_dim: int=0, discrete_column_counts: list=None, header_row: bool=False, label_column: int=-1,  dtype=np.float32):
        self._dtype = dtype

        if isinstance(dataset, str):
            self.xy = torch.from_numpy(np.loadtxt(dataset, delimiter=',', dtype=self._dtype, skiprows=int(header_row)))
        else:
            self.xy = torch.from_numpy(dataset.to_numpy(dtype=self._dtype))
        
        self.numerical_dim = numerical_dim
        self.discrete_column_counts = discrete_column_counts
        self.n_samples = self.xy.shape[0]
        
    def __getitem__(self, index):
        return self.xy[index]

    def __len__(self):
        return self.n_samples
    
class DataSampler(Sampler):
    """
    Parameters
    ----------
    dataset : TabularDataset
        The dataset to sample from. Must have discrete_column_counts attribute.
    sample_size : int, optional
        Total number of samples to draw. If None, uses the entire dataset size.
    batch_size : int, optional
        Size of each batch to return. If None, uses the entire dataset size.
    """
    def __init__(self, dataset: TabularDataset, sample_size: int = None, batch_size: int = None, use_column_pmf: bool = False):

        assert batch_size % 2 == 0, 'batch size must be divisible by 2'

        self.dataset = dataset
        self.num_samples = sample_size if sample_size is not None else len(self.dataset)
        self.batch_size = batch_size if batch_size is not None else len(self.dataset)
        self.use_column_pmf = use_column_pmf

        # Precompute Variables
        self.column_pmf = None
        self.pmfs = []
        self.cumsum_pmfs = []
        self.col_starts = []
        self.col_dims = []
        self.cond_indices = []
        self.cond_indices_lenghts = []
        self.discrete_column_range = np.arange(len(self.dataset.discrete_column_counts)) 
        self.batch_range = np.arange(self.batch_size)  

        n_rows = len(self.dataset)
        start = 0
        self.max_discrete_dim = 0

        # Calculate PMF with log frequency
        for col in self.dataset.discrete_column_counts:
            col_dim = len(col)
            self.col_dims.append(col_dim)
            self.max_discrete_dim = max(col_dim, self.max_discrete_dim)
            self.col_starts.append(start)
            start += col_dim

            log_freqs = {}
            log_freq_sum = 0
            pmf = []

            # Calculate log frequency
            for key, value in col.items():
                log_freq = np.log(value / n_rows + 1e-8)
                log_freq_sum += log_freq
                log_freqs[key] = log_freq

            for key, value in log_freqs.items():
                pmf.append(log_freqs[key] / log_freq_sum)

            self.pmfs.append(pmf)

        # Convert col starts into numpy array
        self.col_starts = np.array(self.col_starts)

        # Convert pmfs into padded numpy array for performance
        self.pmfs_padded = np.zeros((len(self.pmfs), self.max_discrete_dim), dtype=self.dataset._dtype)
        for i, pmf in enumerate(self.pmfs):
            self.pmfs_padded[i, :len(pmf)] = pmf

        # Precompute cumsum of pmfs
        self.cumsum_pmfs = np.cumsum(self.pmfs_padded, axis=1)

        # Use column pmf
        if self.use_column_pmf:
            categories = np.log(np.array([len(col_counts) for col_counts in self.dataset.discrete_column_counts]))
            self.column_pmf = categories / np.sum(categories)

        # Pre-compute valid indices
        for idx in range(sum(self.col_dims)):
            valid_indices = torch.nonzero(self.dataset.xy[:, self.dataset.numerical_dim + idx])[:,0].numpy()
            self.cond_indices.append(valid_indices)
            self.cond_indices_lenghts.append(len(valid_indices))

        # Preallocate cond vector and mask vector
        self.cond_vector = np.zeros((self.batch_size, sum(self.col_dims)), dtype=self.dataset._dtype)
        self.mask_vector = np.zeros((self.batch_size, len(self.col_dims)), dtype=self.dataset._dtype)       

        # Convert valid indices into padded numpy array for performance
        max_indices_length = max(self.cond_indices_lenghts)
        self.cond_indices_padded = np.zeros((len(self.cond_indices), max_indices_length), dtype=int)
        for i, indices in enumerate(self.cond_indices):
            self.cond_indices_padded[i, :len(indices)] = indices

    #TODO: optimize optimize optimize
    def get_samples(self):
        # Select discrete columns
        selected_column_idxs = np.random.choice(self.discrete_column_range, self.batch_size, p=self.column_pmf)

        # Get PMF of selected discrete column
        cumsum_pmfs = self.cumsum_pmfs[selected_column_idxs]

        # Select category based on pmf
        random = np.expand_dims(np.random.rand(cumsum_pmfs.shape[0]), axis=1)
        category_idxs = np.argmax(cumsum_pmfs > random, axis=1)

        column_idxs = self.col_starts[selected_column_idxs] + category_idxs

        self.cond_vector.fill(0.0)
        self.cond_vector[self.batch_range, column_idxs] = 1.0

        self.mask_vector.fill(0.0)
        self.mask_vector[self.batch_range, selected_column_idxs] = 1.0

        random_positions = np.random.rand(len(column_idxs))
        # Random [0, 1) * cond indices count vectorized for performance
        scaled_positions = (random_positions * np.array([self.cond_indices_lenghts[col_idx] for col_idx in column_idxs])).astype(int)
        batch_indices = self.cond_indices_padded[column_idxs, scaled_positions]
        
        return selected_column_idxs, self.mask_vector.copy(), self.cond_vector.copy(), batch_indices
    
    def __iter__(self):
        return self.get_samples()

    def __len__(self):
        return len(self.dataset)
    