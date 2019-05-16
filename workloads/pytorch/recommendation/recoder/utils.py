import numpy as np
from scipy.sparse import coo_matrix


def unzip(l):
  """
  Returns the inverse operation of `zip` on `list`.

  Args:
    l (list): the list to unzip
  """
  return list(map(list, zip(*l)))


def normalize(x, axis=None):
  """
  Returns the normalization of `x` along `axis`.

  Args:
    x (np.array): matrix or vector
    axis (int, optional): the axis along which to compute the normalization
  """
  return x / np.linalg.norm(x, axis=axis).reshape(-1, 1)


def dataframe_to_csr_matrix(dataframe, user_col, item_col,
                            inter_col, item_id_map=None,
                            user_id_map=None):
  """
  Converts a :class:`pandas.DataFrame` of users and items interactions into a :class:`scipy.sparse.csr_matrix`.

  This function returns a tuple of the interactions sparse matrix, a `dict` that maps
  from original item ids in the dataframe to the 0-based item ids, and similarly a `dict` that maps
  from original user ids in the dataframe to the 0-based user ids.

  Args:
    dataframe (pandas.DataFrame): A dataframe containing users and items interactions
    user_col (str): users column name
    item_col (str): items column name
    inter_col (str): user-item interaction value column name
    item_id_map (dict, optional): A dictionary mapping from original item ids into 0-based item ids. If not given,
      the map will be generated using the items column in the dataframe
    user_id_map (dict, optional): A dictionary mapping from original user ids into 0-based user ids. If not given,
      the map will be generated using the users column in the dataframe

  Returns:
    tuple: A tuple of the `csr_matrix`, a :class:`dict` `item_id_map`, and a :class:`dict` `user_id_map`
  """

  if user_id_map is None:
    users = dataframe[user_col].unique()
    user_id_map = {user: userid for userid, user in enumerate(users)}

  if item_id_map is None:
    items = dataframe[item_col].unique()
    item_id_map = {item: itemid for itemid, item in enumerate(items)}

  matrix_size = (len(user_id_map.keys()), len(item_id_map.keys()))

  matrix_users = dataframe[user_col].map(user_id_map)
  matrix_items = dataframe[item_col].map(item_id_map)
  matrix_inters = dataframe[inter_col]

  csr_matrix = coo_matrix((matrix_inters, (matrix_users, matrix_items)), shape=matrix_size).tocsr()

  return csr_matrix, item_id_map, user_id_map
