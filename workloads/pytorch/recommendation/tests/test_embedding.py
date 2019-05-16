from recoder.embedding import AnnoyEmbeddingsIndex

import pytest

import numpy as np



def test_build_index():
  embeddings_mat = np.random.rand(1000, 128)
  index = AnnoyEmbeddingsIndex(embeddings=embeddings_mat)

  index.build(index_file='/tmp/test_embeddings')
  index_loaded = AnnoyEmbeddingsIndex()
  index_loaded.load(index_file='/tmp/test_embeddings')

  assert index_loaded.embedding_size == index.embedding_size and index.embedding_size == 128

  test_item = np.random.randint(1000)
  assert index.get_embedding(test_item) == index_loaded.get_embedding(test_item)

  assert index.get_nns_by_id(test_item, 100) == index_loaded.get_nns_by_id(test_item, 100)

  test_item_1 = np.random.randint(0, 1000)
  test_item_2 = np.random.randint(0, 1000)
  assert index.get_similarity(test_item_1, test_item_2) == \
         index_loaded.get_similarity(test_item_1, test_item_2)

