import annoy as an

import pickle

import glog as log


class EmbeddingsIndex(object):
  """
  An abstract Embeddings Index from which to fetch embeddings and
  execute nearest neighbor search on the items represented by the embeddings

  All ``EmbeddingsIndex`` should implement this interface.
  """

  def get_embedding(self, embedding_id):
    """
    Returns the embedding of the item ``embedding_id``
    """
    raise NotImplementedError

  def get_nns_by_id(self, embedding_id, n):
    """
    Returns the ``n`` nearest neighbors of the ``embedding_id``
    """
    raise NotImplementedError

  def get_nns_by_embedding(self, embedding, n):
    """
    Returns the ``n`` nearest neighbors of the ``embedding``
    """
    raise NotImplementedError

  def get_similarity(self, id1, id2):
    """
    Returns the similarity between item ``id1`` and item ``id2``
    """
    raise NotImplementedError


class AnnoyEmbeddingsIndex(EmbeddingsIndex):
  """
  An ``EmbeddingsIndex`` based on ``AnnoyIndex`` [1] to efficiently execute nearest neighbors
  search with trade off in accuracy.

  The similarity between items is the cosine similarity.

  Args:
    embeddings (numpy.array, optional): the matrix that holds the embeddings of shape
      (number of items * embedding size). Required to build the index.
    id_map (dict, optional): A dict that maps the items original ids to the indices of the embeddings.
      Useful to fetch and do nearest neighbor search on the original items ids. If not provided,
      it will simply be an identity map.
    n_trees (int, optional): n_trees parameter used to build AnnoyIndex.
    search_k (int, optional): search_k parameter used to search the AnnoyIndex for nearest items.
    include_distances (bool, optional): include distances in the result returned on nearest search

  [1]: https://github.com/spotify/annoy
  """

  def __init__(self, embeddings=None, id_map=None,
               n_trees=10, search_k=-1,
               include_distances=False):
    self.embeddings = embeddings
    self.n_trees = n_trees
    self.id_map = id_map
    self.search_k = search_k
    self.include_distances = include_distances

  def build(self, index_file=None):
    """
    Builds the embeddings index, and stores it in ``index_file`` if provided.

    Args:
      index_file (str, optional): the index file path where to save the index. Note: The
        annoy index file is stored in a separate file, which should be in the same directory
        as ``index_file``.
    """
    self.__build_index(index_file=index_file)

  def load(self, index_file):
    """
    Loads the embeddings index from a saved index file.

    Args:
      index_file (str): the index file path to load the state of the index. Note: The
        annoy index file is stored in a separate file, which should be in the same directory
        as index_file.
    """
    self.__load_index(index_file=index_file)

  def __build_index(self, index_file):
    self.embedding_size = self.embeddings.shape[1]

    self.index = an.AnnoyIndex(self.embedding_size, metric='angular')

    for embedding_ind in range(self.embeddings.shape[0]):
      embedding = self.embeddings[embedding_ind, :]
      self.index.add_item(embedding_ind, embedding)

    self.index.build(self.n_trees)

    if self.id_map is None:
      self.id_map = dict([(i, i) for i in range(self.embeddings.shape[0])])

    self.inverse_id_map = dict([(v,k) for k,v in self.id_map.items()])

    if index_file:
      embeddings_file = index_file + '.embeddings'
      state = {
        'embedding_size': self.embedding_size,
        'id_map': self.id_map,
      }

      self.index.save(embeddings_file)
      with open(index_file, 'wb') as _index_file:
        pickle.dump(state, _index_file)

  def __load_index(self, index_file):
    log.info('Loading index file from {}'.format(index_file))
    with open(index_file, 'rb') as _index_file:
      state = pickle.load(_index_file)
    self.embedding_size = state['embedding_size']
    self.id_map = state['id_map']
    embeddings_file = index_file + '.embeddings'
    self.index = an.AnnoyIndex(self.embedding_size, metric='angular')
    self.index.load(embeddings_file)
    self.inverse_id_map = dict([(v,k) for k,v in self.id_map.items()])

  def get_embedding(self, embedding_id):
    return self.index.get_item_vector(self.id_map[embedding_id])

  def get_nns_by_id(self, embedding_id, n):
    nearest_indices = self.index.get_nns_by_item(self.id_map[embedding_id], n, search_k=self.search_k,
                                                 include_distances=self.include_distances)

    if not self.include_distances:
      nearest_ids = [self.inverse_id_map[ind] for ind in nearest_indices]
    else:
      nearest_ids = dict(zip([self.inverse_id_map[ind] for ind in nearest_indices[0]], nearest_indices[1]))

    return nearest_ids

  def get_nns_by_embedding(self, embedding, n):
    nearest_indices = self.index.get_nns_by_vector(embedding, n, search_k=self.search_k,
                                                   include_distances=self.include_distances)

    if not self.include_distances:
      nearest_ids = [self.inverse_id_map[ind] for ind in nearest_indices]
    else:
      nearest_ids = dict(zip([self.inverse_id_map[ind] for ind in nearest_indices[0]], nearest_indices[1]))

    return nearest_ids

  def get_similarity(self, id1, id2):
    distance = self.index.get_distance(self.id_map[id1], self.id_map[id2])
    cosine_similarity = 1 - (distance**2) / 2 # range from -1 to 1
    similarity = (cosine_similarity + 1) / 2 # range from 0 to 1
    return similarity


class MemCacheEmbeddingsIndex(EmbeddingsIndex):
  """
  Caches ``EmbeddingsIndex`` nearest neighbor search results for each item in memory to reduce
  computations.

  Args:
    embedding_index (EmbeddingsIndex): the EmbeddingsIndex to hit on cache misses.
  """

  def __init__(self, embedding_index):
    self.embedding_index = embedding_index # type: EmbeddingsIndex
    self.__nns_cache = {}

  def get_embedding(self, embedding_id):
    return self.embedding_index.get_embedding(embedding_id)

  def get_nns_by_embedding(self, embedding, n):
    return self.embedding_index.get_nns_by_embedding(embedding, n)

  def get_nns_by_id(self, embedding_id, n):
    if embedding_id not in self.__nns_cache:
      self.__nns_cache[embedding_id] = self.embedding_index.get_nns_by_id(embedding_id, n)
    return self.__nns_cache[embedding_id]

  def get_similarity(self, id1, id2):
    return self.embedding_index.get_similarity(id1, id2)
