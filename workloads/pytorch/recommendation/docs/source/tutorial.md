# Tutorial

In this quick tutorial, we will show how to:
- Train an Autoencoder model and a Matrix Factorization model
for implicit feedback collaborative filtering.
- Build your own Factorization model and train it.
- Do negative sampling to speed-up training.
- Evaluate the trained model.

### Training

#### Prepare Data For Training

The data for training/evaluation has to be in a `scipy.sparse.csr_matrix` format.
Typically, the data can loaded as a `pandas.DataFrame`, that can be converted into
a sparse matrix with `recoder.utils.dataframe_to_csr_matrix`.


```python
import pickle

import pandas as pd
from scipy.sparse import save_npz

from recoder.utils import dataframe_to_csr_matrix


# train_df is a dataframe where each row is a user-item interaction
# and the value of that interaction
train_df = pd.read_csv('train.csv')

train_matrix, item_id_map, user_id_map = dataframe_to_csr_matrix(train_df,
                                                                 user_col='user',
                                                                 item_col='item',
                                                                 inter_col='score')

# train_matrix is a user by item interactions matrix

# item_id_map maps the original item ids into indexed item ids, such that
# the interactions with item 'whatever' can be retrieved with
# train_matrix[:, item_id_map['whatever']]

# user_id_map is like item_id_map but for users. The interactions of user 'whoever'
# can be retrieved with train_matrix[user_id_map['whoever'], :]


# you can save the sparse matrix so you don't have to
# get the sparse matrix everytime
save_npz('train.npz', matrix=train_matrix)

# also better saving the item_id_map and user_id_map
# you can do that with pickle
with open('item_id_map.dict', 'wb') as _item_id_map_file_pointer:
  pickle.dump(item_id_map, _item_id_map_file_pointer)

with open('user_id_map.dict', 'wb') as _user_id_map_file_pointer:
  pickle.dump(user_id_map, _user_id_map_file_pointer)
```


#### Autoencoder Model

```python
from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder

import scipy.sparse as sparse

# Load the training sparse matrix
train_matrix = sparse.load_npz('train.npz')

train_dataset = RecommendationDataset(train_matrix)

# Define your model
model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                           noise_prob=0.5, sparse=True)

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, negative_sampling=True)
```

#### Matrix Factorization Model

Same as training Autoencoder model, just replace the Autoencoder with a Matrix Factorization
Model.

```python
from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import MatrixFactorization

import scipy.sparse as sparse

# Load the training sparse matrix
train_matrix = sparse.load_npz('train.npz')

train_dataset = RecommendationDataset(train_matrix)

# Define your model
model = MatrixFactorization(embedding_size=200, activation_type='tanh',
                            dropout_prob=0.5, sparse=True)

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, negative_sampling=True)
```


#### Your Own Factorization Model

If you want to build your own Factorization model with the objective
of reconstructing the interactions matrix, all you have to do is implement
``recoder.nn.FactorizationModel`` interface.

```python
from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import FactorizationModel

import scipy.sparse as sparse


# Implement your model
class YourModel(FactorizationModel):

  def init_model(self, num_items=None, num_users=None):
    # Initializes your model with the number of items and users.
    pass

  def model_params(self):
    # Returns your model parameters in a dict.
    # Used by Recoder when saving the model.
    pass

  def load_model_params(self, model_params):
    # Loads the model parameters into the model.
    # Used by Recoder when loading the model from a snapshot.
    pass

  def forward(self, input, input_users=None,
              input_items=None, target_users=None,
              target_items=None):
    # A forward pass on the model
    # input_users are the users in the input batch
    # input_items are the items in the input batch
    # target_items are the items to be predicted
    pass


# Load the training sparse matrix
train_matrix = sparse.load_npz('train.npz')

train_dataset = RecommendationDataset(train_matrix)

# Define your model
model = YourModel()

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, negative_sampling=True)
```

#### Save your model

```python
# You can save your model while training at different epoch checkpoints using
# model_checkpoint_prefix and checkpoint_freq params

# model state file prefix that will be appended by epoch number
model_checkpoint_prefix = 'models/model_'

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, negative_sampling=True,
              model_checkpoint_prefix=model_checkpoint_prefix,
              checkpoint_freq=10)

# or you can directly call recoder.save_state
recoder.save_state(model_checkpoint_prefix)
```

#### Continue training

```python
from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder

import scipy.sparse as sparse

# Load the training sparse matrix
train_matrix = sparse.load_npz('train.npz')

train_dataset = RecommendationDataset(train_matrix)

# your saved model
model_file = 'models/your_model'

# Initialize your model
# No need to set model parameters since they will be loaded
# when initializing Recoder from a saved model
model = DynamicAutoencoder()


# Initialize Recoder
recoder = Recoder(model=model, use_cuda=True)
recoder.init_from_model_file(model_file)

recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, negative_sampling=True)
```

#### Tips

Recoder supports training with sparse gradients. Sparse gradients training is only
supported currently by the ``torch.optim.SparseAdam`` optimizer. This is specially helpful
for training big embedding layers such as the users and items embedding
layers in the Autoencoder and MatrixFactorization models. Set the ``sparse`` parameter
in ``Autoencoder`` and ``MatrixFactorization`` to ``True`` in order to return sparse gradients
and this can lead to 1.5-2x training speed-up. If you want to build your own model and have
the embedding layers return sparse gradients, ``Recoder`` should be able to detect that.


### Mini-batch based negative sampling

Mini-batch based negative sampling is based on the simple idea of sampling, for each
user, only the negative items that the other users in the mini-batch have interacted
with. This sampling procedure is biased toward popular items and in order to tune the
sampling probability of each negative item, one has to tune the training batch-size.
Mini-batch based negative sampling can speed-up training by 2-4x while having a small
drop in recommendation performance.

- To use mini-batch based negative sampling, you have to set ``negative_sampling`` to ``True`` in
``Recoder.train`` and tune it with the ``batch_size``:

```python
recoder.train(train_dataset=train_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, negative_sampling=True)
```

- For large datasets with large number of items, we need a large
number of negative samples, hence a large batch size, which makes
the batch not fit into memory and expensive to train on. In that case,
we can simply generate the sparse batch with a large batch size and
then slice it into smaller batches, and train on the small batches.
To do this you can fix the ``batch_size`` to a specific value, and
instead tune the ``num_sampling_users`` in order to increase the number
of negative samples.

```python
recoder.train(train_dataset=train_dataset, batch_size=500,
              negative_sampling=True, num_sampling_users=2000, lr=1e-3,
              weight_decay=2e-5, num_epochs=100, num_data_workers=4)
```

### Evaluation

You can evaluate your model with different metrics. Currently, there
are 3 metrics implemented: Recall, NDCG, and Average Precision. You can
also implement your own ``recoder.metrics.Metric``.

#### Evaluating your model while training

```python
from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder
from recoder.metrics import AveragePrecision, Recall, NDCG

import scipy.sparse as sparse


# Load the training sparse matrix
train_matrix = sparse.load_npz('train.npz')

# validation set. Split your val set into two splits.
# One split will be used as input to the model to
# generate predictions, and the other is which the
# model predictions will be evaluated on
val_input_matrix = sparse.load_npz('test_input.npz')
val_target_matrix = sparse.load_npz('test_target.npz')

train_dataset = RecommendationDataset(train_matrix)

val_dataset = RecommendationDataset(val_input_matrix, val_target_matrix)


# Define your model
model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                           noise_prob=0.5, sparse=True)

# Initialize your metrics
metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

# Recoder takes a factorization model and trains it
recoder = Recoder(model=model, use_cuda=True,
                  optimizer_type='adam', loss='logistic')

recoder.train(train_dataset=train_dataset,
              val_dataset=val_dataset, batch_size=500,
              lr=1e-3, weight_decay=2e-5, num_epochs=100,
              num_data_workers=4, negative_sampling=True,
              metrics=metrics, eval_num_recommendations=100,
              eval_freq=5)
```

#### Evaluating your model after training

```python
from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.nn import DynamicAutoencoder
from recoder.metrics import AveragePrecision, Recall, NDCG

import scipy.sparse as sparse


# validation set. Split your val set into two splits.
# One split will be used as input to the model to
# generate predictions, and the other is which the
# model predictions will be evaluated on
test_input_matrix = sparse.load_npz('test_input.npz')
test_target_matrix = sparse.load_npz('test_target.npz')

test_dataset = RecommendationDataset(test_input_matrix, test_target_matrix)

# your saved model
model_file = 'models/your_model'

# Initialize your model
# No need to set model parameters since they will be loaded
# when initializing Recoder from a saved model
model = DynamicAutoencoder()

# Initialize your metrics
metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

# Initialize Recoder
recoder = Recoder(model=model, use_cuda=True)
recoder.init_from_model_file(model_file)

# Evaluate on the top 100 recommendations
num_recommendations = 100

recoder.evaluate(eval_dataset=test_dataset, num_recommendations=num_recommendations,
                 metrics=metrics, batch_size=500)
```
