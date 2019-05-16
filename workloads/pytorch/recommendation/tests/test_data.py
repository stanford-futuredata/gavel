from recoder.data import RecommendationDataset, RecommendationDataLoader, BatchCollator
from recoder.utils import dataframe_to_csr_matrix

import pandas as pd
import numpy as np
import torch

import pytest


def generate_dataframe():
  data = pd.DataFrame()
  data['user'] = np.random.randint(0, 100, 1000)
  data['item'] = np.random.randint(0, 200, 1000)
  data['inter'] = np.ones(1000)
  data = data.drop_duplicates(['user', 'item']).reset_index(drop=True)
  return data


@pytest.fixture
def input_dataframe():
  return generate_dataframe()


@pytest.fixture
def target_dataframe():
  return generate_dataframe()


def test_RecommendationDataset(input_dataframe):
  interactions_matrix, item_id_map, user_id_map = dataframe_to_csr_matrix(input_dataframe, user_col='user',
                                                                          item_col='item', inter_col='inter')

  dataset = RecommendationDataset(interactions_matrix)

  assert len(dataset) == len(np.unique(input_dataframe['user']))

  replica_df = pd.DataFrame(input_dataframe)

  for index in range(len(dataset)):
    user_interactions, _ = dataset[index]
    user = user_interactions.users[0]
    assert user_interactions.interactions_matrix.getnnz() == len(replica_df[replica_df.user.map(user_id_map) == user])

    for item_id, inter_val in zip(user_interactions.interactions_matrix.nonzero()[1],
                                  user_interactions.interactions_matrix.data):
      assert len(replica_df[(replica_df.user.map(user_id_map) == user)
                            & (replica_df.item.map(item_id_map) == item_id)
                            & (replica_df.inter == inter_val)]) > 0
      replica_df = replica_df[~ ((replica_df.user.map(user_id_map) == user)
                                & (replica_df.item.map(item_id_map) == item_id)
                                & (replica_df.inter == inter_val))]

    assert user_interactions.interactions_matrix.getnnz() > 0

  # check that both the returned list of interactions and the dataframe contain
  # the same of interactions
  assert len(replica_df) == 0


def test_RecommendationDataset_target(input_dataframe, target_dataframe):
  common_users = input_dataframe.merge(target_dataframe, how='inner', on='user').user.unique()
  common_items = input_dataframe.merge(target_dataframe, how='inner', on='item').item.unique()

  input_dataframe = input_dataframe[input_dataframe.user.isin(common_users)
                                    & input_dataframe.item.isin(common_items)]
  target_dataframe = target_dataframe[target_dataframe.user.isin(common_users)
                                      & target_dataframe.item.isin(common_items)]

  interactions_matrix, item_id_map, user_id_map = dataframe_to_csr_matrix(input_dataframe, user_col='user',
                                                                          item_col='item', inter_col='inter')

  target_interactions_matrix, _, _ = dataframe_to_csr_matrix(target_dataframe, user_col='user',
                                                             item_col='item', inter_col='inter',
                                                             item_id_map=item_id_map, user_id_map=user_id_map)

  dataset = RecommendationDataset(interactions_matrix, target_interactions_matrix)

  test_index = np.random.randint(0, len(dataset))

  input_interactions, target_interactions = dataset[test_index]

  assert input_interactions.users == target_interactions.users

  assert input_interactions.interactions_matrix.getnnz() > 0 \
         and target_interactions.interactions_matrix.getnnz() > 0


@pytest.mark.parametrize("batch_size,num_sampling_users",
                         [(5, 0),
                          (5, 10)])
def test_RecommendationDataLoader(input_dataframe, target_dataframe,
                                  batch_size, num_sampling_users):
  common_users = input_dataframe.merge(target_dataframe, how='inner', on='user').user.unique()
  common_items = input_dataframe.merge(target_dataframe, how='inner', on='item').item.unique()

  input_dataframe = input_dataframe[input_dataframe.user.isin(common_users)
                                    & input_dataframe.item.isin(common_items)]
  target_dataframe = target_dataframe[target_dataframe.user.isin(common_users)
                                      & target_dataframe.item.isin(common_items)]

  interactions_matrix, item_id_map, user_id_map = dataframe_to_csr_matrix(input_dataframe, user_col='user',
                                                                          item_col='item', inter_col='inter')

  target_interactions_matrix, _, _ = dataframe_to_csr_matrix(target_dataframe, user_col='user',
                                                             item_col='item', inter_col='inter',
                                                             item_id_map=item_id_map, user_id_map=user_id_map)

  dataset = RecommendationDataset(interactions_matrix, target_interactions_matrix)

  dataloader = RecommendationDataLoader(dataset, batch_size=batch_size,
                                        negative_sampling=True,
                                        num_sampling_users=num_sampling_users)

  for batch_idx, (input, target) in enumerate(dataloader, 1):
    input_idx, input_val, input_size, input_items = input.indices, input.values, input.size, input.items
    input_dense = torch.sparse.FloatTensor(input_idx, input_val, input_size).to_dense()

    target_idx, target_val, target_size, target_words = target.indices, target.values, target.size, target.items
    target_dense = torch.sparse.FloatTensor(target_idx, target_val, target_size).to_dense()

    assert target is not None

    assert input_dense.size(0) == batch_size \
           or batch_idx == len(dataloader) and input_dense.size(0) == len(dataset) % batch_size
    assert input_dense.size(1) == len(input_items)


@pytest.mark.parametrize("batch_size",
                         [1, 2, 5, 10, 13])
def test_BatchCollator(input_dataframe, batch_size):
  interactions_matrix, item_id_map, user_id_map = dataframe_to_csr_matrix(input_dataframe, user_col='user',
                                                                          item_col='item', inter_col='inter')

  dataset = RecommendationDataset(interactions_matrix)

  batch_collator = BatchCollator(batch_size=batch_size,
                                 negative_sampling=True)

  big_batch, _ = dataset[np.arange(len(dataset))]

  batches = batch_collator.collate(big_batch)

  assert len(batches) == np.ceil(len(dataset) / batch_size)

  current_batch = 0
  for batch in batches:
    input_idx, input_val, input_size, input_words = batch.indices, batch.values, batch.size, batch.items
    input_dense = torch.sparse.FloatTensor(input_idx, input_val, input_size).to_dense()

    batch_users = big_batch.users[current_batch:current_batch+batch_size]
    batch_sparse_matrix = big_batch.interactions_matrix[current_batch:current_batch+batch_size]

    num_values_per_user = [batch_sparse_matrix[i].getnnz() for i in range(len(batch_users))]

    assert (input_dense > 0).float().sum(dim=1).tolist() == num_values_per_user

    item_idx_map = {item_id:item_idx for item_idx, item_id in enumerate(input_words.tolist())}

    for user_idx in range(len(batch_users)):
      for item_id, val in zip(batch_sparse_matrix[user_idx].nonzero()[1], batch_sparse_matrix[user_idx].data):
        assert item_id in input_words
        assert input_dense[user_idx, item_idx_map[item_id]] == val

    current_batch += batch_size
