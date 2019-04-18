import torch
from torch import nn
import torch.nn.functional as F


def activation(x, act):
  if act == 'none': return x
  func = getattr(torch, act)
  return func(x)


class FactorizationModel(nn.Module):
  """
  Base class for factorization models. All subclasses should implement
  the following methods.
  """

  def init_model(self, num_items=None, num_users=None):
    """
    Initializes the model with the number of users and items to be represented.

    Args:
      num_users (int): number of users to be represented in the model
      num_items (int): number of items to be represented in the model
    """
    raise NotImplementedError

  def model_params(self):
    """
    Returns the model parameters. Mainly used when storing the model hyper-parameters
    (i.e hidden layers, activation..etc) in a snapshot file by :class:`recoder.model.Recoder`.

    Returns:
      dict: Model parameters.
    """
    raise NotImplementedError

  def load_model_params(self, model_params):
    """
    Loads the ``model_params`` into the model. Mainly used when loading the model
    hyper-parameters (i.e hidden layers, activation..etc) from a snapshot file of
    the model stored by :class:`recoder.model.Recoder`.

    Args:
      model_params (dict): model parameters
    """
    raise NotImplementedError

  def forward(self, input, input_users=None,
              input_items=None, target_users=None,
              target_items=None):
    """
    Applies a forward pass of the input on the latent factor model.

    Args:
      input (torch.FloatTensor): the input dense matrix of user item interactions.
      input_users (torch.LongTensor): the users represented in the input batch, where
        each user corresponds to a row in ``input`` based on their index.
      input_items (torch.LongTensor): the items represented in the input batch, where
        each items corresponds to a column in ``input`` based on their index.
      target_users (torch.LongTensor): the target users to predict. Typically, this is not used,
        but kept for consistency.
      target_items (torch.LongTensor): the target items to predict.
    """
    raise NotImplementedError


class DynamicAutoencoder(FactorizationModel):
  """
  An Autoencoder module that processes variable size vectors. This is
  particularly efficient for cases where we only want to reconstruct sub-samples
  of a large sparse vector and not the whole vector, i.e negative sampling.

  Let `F` be a `DynamicAutoencoder` function that reconstructs vectors of size `d`,
  let `X` be a matrix of size `Bxd` where `B` is the batch size, and
  let `Z` be a sub-matrix of `X` and `I` be a vector of any length, such that `1 <= I[i] <= d`
  and `Z = X[:, I]`. The reconstruction of `Z` is `F(Z, I)`. See `Examples`.

  Args:
    hidden_layers (list): autoencoder hidden layers sizes. only the encoder layers.
    activation_type (str, optional): activation function to use for hidden layers.
      all activations in torch.nn.functional are supported
    is_constrained (bool, optional): constraining model by using the encoder weights in the
      decoder (tying the weights).
    dropout_prob (float, optional): dropout probability at the bottleneck layer
    noise_prob (float, optional): dropout (noise) probability at the input layer
    sparse (bool, optional): if True, gradients w.r.t. to the embedding layers weight matrices
      will be sparse tensors. Currently, sparse gradients are only fully supported by
      ``torch.optim.SparseAdam``.

  Examples::

    >>>> autoencoder = DynamicAutoencoder([500,100])
    >>>> batch_size = 32
    >>>> input = torch.rand(batch_size, 5)
    >>>> input_items = torch.LongTensor([10, 126, 452, 29, 34])
    >>>> output = autoencoder(input, input_items=input_items, target_items=input_items)
    >>>> output
       0.0850  0.9490  ...   0.2430  0.5323
       0.3519  0.4816  ...   0.9483  0.2497
            ...         ⋱         ...
       0.8744  0.8194  ...   0.5755  0.2090
       0.5006  0.9532  ...   0.8333  0.4330
      [torch.FloatTensor of size 32x5]
    >>>>
    >>>> # predicting a different target of items
    >>>> target_items = torch.LongTensor([31, 14, 95, 49, 10, 36, 239])
    >>>> output = autoencoder(input, input_items=input_items, target_items=target_items)
    >>>> output
       0.5446  0.5468  ...   0.9854  0.6465
       0.0564  0.1238  ...   0.5645  0.6576
            ...         ⋱         ...
       0.0498  0.6978  ...   0.8462  0.2135
       0.6540  0.5686  ...   0.6540  0.4330
      [torch.FloatTensor of size 32x7]
    >>>>
    >>>> # reconstructing the whole vector
    >>>> input = torch.rand(batch_size, 500)
    >>>> output = autoencoder(input)
    >>>> output
       0.0865  0.9054  ...   0.8987  0.0456
       0.9852  0.6540  ...   0.1205  0.8488
            ...         ⋱         ...
       0.4650  0.3540  ...   0.5646  0.5605
       0.6940  0.2140  ...   0.9820  0.5405
      [torch.FloatTensor of size 32x500]
  """

  def __init__(self, hidden_layers=None, activation_type='tanh',
               is_constrained=False, dropout_prob=0.0,
               noise_prob=0.0, sparse=False):
    super().__init__()
    self.activation_type = activation_type
    self.is_constrained = is_constrained
    self.hidden_layers = hidden_layers
    self.dropout_prob = dropout_prob
    self.noise_prob = noise_prob
    self.sparse = sparse

    self.num_items = None
    self.num_embeddings = None
    self.noise_layer = None
    self.dropout_layer = None

  def init_model(self, num_items=None, num_users=None):
    self.num_items = num_items
    self.num_embeddings = num_items

    self.__create_encoding_layers()
    self.__create_decoding_layers()

    self.noise_layer = None
    if self.noise_prob > 0.0:
      self.noise_layer = nn.Dropout(p=self.noise_prob)

    self.dropout_layer = None
    if self.dropout_prob > 0.0:
      self.dropout_layer = nn.Dropout(p=self.dropout_prob)

    if self.is_constrained:
      self.__tie_weights()

  def model_params(self):
    return {
      'hidden_layers': self.hidden_layers,
      'activation_type': self.activation_type,
      'is_constrained': self.is_constrained,
      'dropout_prob': self.dropout_prob,
      'noise_prob': self.noise_prob
    }

  def load_model_params(self, model_params):
    self.hidden_layers = model_params['hidden_layers']
    self.activation_type = model_params['activation_type']
    self.is_constrained = model_params['is_constrained']
    self.dropout_prob = model_params['dropout_prob']
    self.noise_prob = model_params['noise_prob']

  def __create_encoding_layers(self):
    self.en_embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_layers[0],
                                           sparse=self.sparse)

    self.__en_linear_embedding_layer = LinearEmbedding(self.en_embedding_layer, input_based=True)
    self.encoding_layers = nn.Sequential(*self.__create_coding_layers(self.hidden_layers))

    nn.init.xavier_uniform_(self.en_embedding_layer.weight)
    nn.init.constant_(self.__en_linear_embedding_layer.bias, 0)

  def __create_decoding_layers(self):
    _decoding_layers = self.__create_coding_layers(list(reversed(self.hidden_layers)))
    if self.is_constrained:
      for ind, decoding_layer in enumerate(_decoding_layers):
        # Deleting layer weight to unregister it as a parameter
        # Only register decoding layers biases as parameters
        del decoding_layer.weight

      # Reset the decoding layers weights as encoding layers weights tranpose
      # These won't be registered as model parameters
      for el, dl in zip(self.encoding_layers, reversed(_decoding_layers)):
        dl.weight = el.weight.t()

      self.de_embedding_layer = self.en_embedding_layer
    else:
      self.de_embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_layers[0],
                                             sparse=self.sparse)

    self.decoding_layers = nn.Sequential(*_decoding_layers)

    self.__de_linear_embedding_layer = LinearEmbedding(self.de_embedding_layer, input_based=False)

    nn.init.xavier_uniform_(self.de_embedding_layer.weight)
    nn.init.constant_(self.__de_linear_embedding_layer.bias, 0)

  def __create_coding_layers(self, layer_sizes):
    layers = []
    for ind, layer_size in enumerate(layer_sizes[1:], 1):
      layer = nn.Linear(layer_sizes[ind-1], layer_size)
      layers.append(layer)
      torch.nn.init.xavier_uniform_(layer.weight)
      torch.nn.init.constant_(layer.bias, 0)

    return layers

  def __tie_weights(self):
    for el, dl in zip(self.encoding_layers, reversed(self.decoding_layers)):
      dl.weight = el.weight.t()

  def forward(self, input, input_users=None,
              input_items=None, target_users=None,
              target_items=None):
    if self.is_constrained:
      self.__tie_weights()

    # Normalize the input
    z = F.normalize(input, p=2, dim=1)
    if self.noise_prob > 0.0:
      z = self.noise_layer(z)

    z = self.__en_linear_embedding_layer(input_items, z)
    z = activation(z, self.activation_type)

    for encoding_layer in self.encoding_layers:
      z = activation(encoding_layer(z), self.activation_type)

    if self.dropout_prob > 0.0:
      z = self.dropout_layer(z)

    for decoding_layer in self.decoding_layers:
      z = activation(decoding_layer(z), self.activation_type)

    z = self.__de_linear_embedding_layer(target_items, z)

    return z


class LinearEmbedding(nn.Module):

  def __init__(self, embedding_layer: nn.Embedding, input_based=True, bias=True):
    super().__init__()
    self.embedding_layer = embedding_layer
    self.input_based = input_based
    self.in_features = embedding_layer.num_embeddings if input_based else embedding_layer.embedding_dim
    self.out_features = embedding_layer.embedding_dim if input_based else embedding_layer.num_embeddings
    if bias:
      self.bias = nn.Parameter(torch.Tensor(self.out_features))
    else:
      self.bias = None

  def forward(self, x, y):
    if x is not None:
      _weight = self.embedding_layer(x)
      _bias = self.bias if self.input_based else self.bias.index_select(0, x)
    else:
      _weight = self.embedding_layer.weight
      _bias = self.bias

    if self.input_based:
      return F.linear(y, _weight.t(), _bias)
    else:
      return F.linear(y, _weight, _bias)


class MatrixFactorization(FactorizationModel):
  """
  Defines a Matrix Factorization model for collaborative filtering. This is
  particularly efficient for cases where we only want to reconstruct sub-samples
  of a large sparse vector and not the whole vector, i.e negative sampling.

  Args:
    embedding_size (int): embedding size (rank) of the latent factors of users and items
    activation_type (str, optional): activation function to be applied on the user embedding.
      all activations in torch.nn.functional are supported.
    dropout_prob (float, optional): dropout probability to be applied on the user embedding
    sparse (bool, optional): if True, gradients w.r.t. to the embedding layers weight matrices
      will be sparse tensors. Currently, sparse gradients are only fully supported by
      ``torch.optim.SparseAdam``.

  """
  def __init__(self, embedding_size, activation_type='none',
               dropout_prob=0, sparse=False):
    super().__init__()
    self.embedding_size = embedding_size
    self.activation_type = activation_type
    self.dropout_prob = dropout_prob

    self.num_users = None
    self.num_items = None
    self.user_embedding_layer = None
    self.item_embedding_layer = None
    self.bias = None
    self.dropout_layer = None
    self.sparse = sparse

  def init_model(self, num_items=None, num_users=None):
    self.num_users = num_users
    self.num_items = num_items

    self.user_embedding_layer = nn.Embedding(self.num_users, self.embedding_size,
                                             sparse=self.sparse)
    self.item_embedding_layer = nn.Embedding(self.num_items, self.embedding_size,
                                             sparse=self.sparse)
    self.bias = nn.Parameter(torch.Tensor(self.num_items))

    self.dropout_layer = None
    if self.dropout_prob > 0.0:
      self.dropout_layer = nn.Dropout(p=self.dropout_prob)

    nn.init.xavier_uniform_(self.user_embedding_layer.weight)
    nn.init.xavier_uniform_(self.item_embedding_layer.weight)
    nn.init.constant_(self.bias, 0)

  def model_params(self):
    return {
      'embedding_size': self.embedding_size,
      'activation_type': self.activation_type,
      'dropout_prob': self.dropout_prob,
    }

  def load_model_params(self, model_params):
    self.embedding_size = model_params['embedding_size']
    self.activation_type = model_params['activation_type']
    self.dropout_prob = model_params['dropout_prob']

  def forward(self, input, input_users=None,
              input_items=None, target_users=None,
              target_items=None):

    users_embeddings = self.user_embedding_layer(input_users)
    users_embeddings = activation(users_embeddings, self.activation_type)

    if self.dropout_prob > 0:
      users_embeddings = self.dropout_layer(users_embeddings)

    if target_items is None:
      items_embeddings = self.item_embedding_layer.weight
      bias = self.bias
    else:
      items_embeddings = self.item_embedding_layer(target_items)
      bias = self.bias.index_select(0, target_items)

    output = F.linear(users_embeddings, items_embeddings, bias)
    return output
