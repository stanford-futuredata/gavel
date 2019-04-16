import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.sputils as sputils

import recoder.utils as utils


CSR_MATRIX_INDEX_SIZE_LIMIT = 2000


class UsersInteractions:
  """
  Holds the interactions of a set of users in an interactions sparse matrix

  Args:
    users (np.array): users being represented.
    interactions_matrix (scipy.sparse.csr_matrix): user-item interactions matrix, where ``interactions_matrix[i]``
      correspond to the interactions of ``users[i]``.
  """
  def __init__(self, users, interactions_matrix):
    self.users = users
    self.interactions_matrix = interactions_matrix


class RecommendationDataset(Dataset):
  """
  Represents a :class:`torch.utils.data.Dataset` that iterates through the users interactions with items.

  Indexing this dataset returns a :class:`UsersInteractions` containing the interactions
  of the users in the index.

  Args:
    interactions_matrix (scipy.sparse.csr_matrix): the user-item interactions matrix.
    target_interactions_matrix (scipy.sparse.csr_matrix, optional): the target user-item interactions
      matrix. Mainly used for evaluation, representing the items to recommend.
  """

  def __init__(self, interactions_matrix, target_interactions_matrix=None):
    self.interactions_matrix = interactions_matrix  # type: sparse.csr_matrix
    self.target_interactions_matrix = target_interactions_matrix  # type: sparse.csr_matrix
    self.users = np.arange(self.interactions_matrix.shape[0])
    self.items = np.arange(self.interactions_matrix.shape[1])

  def __len__(self):
    return self.interactions_matrix.shape[0]

  def __getitem__(self, index):
    assert sputils.issequence(index) or sputils.isintlike(index)

    users = np.array(index).reshape(-1,)

    extracted_sparse_matrix = self._extract(self.interactions_matrix, index)

    if self.target_interactions_matrix is None:
      return UsersInteractions(users=users, interactions_matrix=extracted_sparse_matrix), None
    else:
      extracted_target_sparse_matrix = self._extract(self.target_interactions_matrix, index)
      return UsersInteractions(users=users, interactions_matrix=extracted_sparse_matrix), \
             UsersInteractions(users=users, interactions_matrix=extracted_target_sparse_matrix)

  def _extract(self, sparse_matrix, index):

    if sputils.issequence(index) and len(index) > CSR_MATRIX_INDEX_SIZE_LIMIT:
      # It happens that scipy implements the indexing of a csr_matrix with a list using
      # matrix multiplication, which gets to be an issue if the size of the index list is
      # large and lead to memory issues
      # Reference: https://stackoverflow.com/questions/46034212/sparse-matrix-slicing-memory-error/46040827#46040827

      # In order to solve this issue, simply chunk the index into smaller indices of
      # size CSR_MATRIX_INDEX_SIZE_LIMIT and then stack the extracted chunks

      sparse_matrix_slices = []
      for offset in range(0, len(index), CSR_MATRIX_INDEX_SIZE_LIMIT):
        sparse_matrix_slices.append(sparse_matrix[index[offset: offset + CSR_MATRIX_INDEX_SIZE_LIMIT]])

      extracted_sparse_matrix = sparse.vstack(sparse_matrix_slices)
    else:
      extracted_sparse_matrix = sparse_matrix[index]

    return extracted_sparse_matrix


class RecommendationDataLoader:
  """
  A ``DataLoader`` similar to ``torch.utils.data.DataLoader`` that handles
  :class:`RecommendationDataset` and generate batches with negative sampling.

  By default, if no ``collate_fn`` is provided, the :func:`BatchCollator.collate` will
  be used, and iterating through this dataloader will return a :class:`Batch` at each
  iteration.

  Args:
    dataset (RecommendationDataset): dataset from which to load the data
    batch_size (int): number of samples per batch
    negative_sampling (bool, optional): whether to apply mini-batch based negative sampling or not.
    num_sampling_users (int, optional): number of users to consider for mini-batch based negative
      sampling. This is useful for increasing the number of negative samples while keeping the
      batch-size small. If 0, then num_sampling_users will be equal to batch_size.
    num_workers (int, optional): how many subprocesses to use for data loading.
    collate_fn (callable, optional): A function that transforms a :class:`UsersInteractions` into
      a mini-batch.
  """
  def __init__(self, dataset, batch_size, negative_sampling=False,
               num_sampling_users=0, num_workers=0, collate_fn=None):
    self.dataset = dataset # type: RecommendationDataset
    self.num_sampling_users = num_sampling_users
    self.num_workers = num_workers
    self.batch_size = batch_size
    self.negative_sampling = negative_sampling

    if self.num_sampling_users == 0:
      self.num_sampling_users = batch_size

    assert self.num_sampling_users >= batch_size, 'num_sampling_users should be at least equal to the batch_size'

    self.batch_collator = BatchCollator(batch_size=self.batch_size, negative_sampling=self.negative_sampling)

    # Wrapping a BatchSampler within a BatchSampler
    # in order to fetch the whole mini-batch at once
    # from the dataset instead of fetching each sample on its own
    batch_sampler = BatchSampler(BatchSampler(RandomSampler(dataset),
                                              batch_size=self.num_sampling_users, drop_last=False),
                                 batch_size=1, drop_last=False)

    if collate_fn is None:
      self._collate_fn = self.batch_collator.collate
      self._use_default_data_generator = True
    else:
      self._collate_fn = collate_fn
      self._use_default_data_generator = False

    self._dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                  num_workers=num_workers, collate_fn=self._collate)

  def _default_data_generator(self):
    for input, target in self._dataloader:
      for batch_ind in range(len(input)):
        if target is None:
          yield input[batch_ind], None
        else:
          yield input[batch_ind], target[batch_ind]

  def _collate(self, batch):
    _input_batch, _target_batch = utils.unzip(batch)

    # _input_batch is a list of size 1, where the only
    # element is the UsersInteractions batch
    input = self._collate_fn(_input_batch[0])

    if _target_batch[0] is None:
      target = None
    else:
      target = self._collate_fn(_target_batch[0])

    return input, target

  def __iter__(self):
    if self._use_default_data_generator:
      return self._default_data_generator()

    return self._dataloader.__iter__()

  def __len__(self):
    return int(np.ceil(len(self.dataset) / self.batch_collator.batch_size))


class Batch:
  """
  Represents a sparse batch of users and items interactions.

  Args:
    users (torch.LongTensor): users that are in the batch
    items (torch.LongTensor): items that are in the batch
    indices (torch.LongTensor): the indices of the interactions in the sparse matrix
    values (torch.LongTensor): the values of the interactions
    size (torch.Size): the size of the sparse interactions matrix
  """
  def __init__(self, users, items,
               indices, values, size):
    self.users = users
    self.items = items
    self.indices = indices
    self.values = values
    self.size = size


class BatchCollator:
  """
  Collator of :class:`UsersInteractions`. It collates the users interactions into multiple :class:`Batch`
  based on ``batch_size``.

  Args:
    batch_size (int): number of samples per batch
    negative_sampling (bool, optional): whether to apply mini-batch based negative sampling or not.
  """
  def __init__(self, batch_size, negative_sampling=False):
    self.batch_size = batch_size
    self.negative_sampling = negative_sampling

  def collate(self, users_interactions):
    """
    Collates :class:`UsersInteractions` into batches of size ``batch_size``.

    Args:
      users_interactions (UsersInteractions): a :class:`UsersInteractions`.

    Returns:
      list[Batch]: list of batches.
    """
    batch_users = users_interactions.users

    users_inds, items_inds = users_interactions.interactions_matrix.nonzero()
    if self.negative_sampling:
      # The positive item ids in the batch
      # This is simply equivalent to only selecting the non-zero columns
      # in the sparse matrix
      batch_items, items_inds = np.unique(items_inds, return_inverse=True)

      vector_dim = len(batch_items)
      batch_items = torch.LongTensor(batch_items)
    else:
      vector_dim = users_interactions.interactions_matrix.shape[1]
      batch_items = None

    batch_users = torch.LongTensor(batch_users)
    slices = []
    current_ind = 0
    for offset in range(0, users_interactions.interactions_matrix.shape[0], self.batch_size):
      slice_sparse_matrix = users_interactions.interactions_matrix[offset: offset + self.batch_size]

      slice_batch_users = batch_users[offset: offset + self.batch_size]

      slice_users_inds = slice_sparse_matrix.nonzero()[0]

      num_nnz = slice_sparse_matrix.getnnz()
      slice_items_inds = items_inds[current_ind:current_ind+num_nnz]
      current_ind += num_nnz

      slice_inter_vals = slice_sparse_matrix.data

      indices = torch.LongTensor([slice_users_inds, slice_items_inds])
      values = torch.FloatTensor(slice_inter_vals)

      slices.append(Batch(items=batch_items, users=slice_batch_users,
                          indices=indices, values=values,
                          size=torch.Size([slice_sparse_matrix.shape[0], vector_dim])))

    return slices
