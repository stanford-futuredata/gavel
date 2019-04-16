from recoder.nn import DynamicAutoencoder

import torch

import pytest


@pytest.fixture
def autoencoder():
  autoencoder = DynamicAutoencoder([300, 200])
  autoencoder.init_model(num_items=500)
  return autoencoder


def test_DynamicAutoencoder(autoencoder):
  assert autoencoder.en_embedding_layer.embedding_dim == 300
  assert autoencoder.de_embedding_layer.embedding_dim == 300

  assert len(autoencoder.encoding_layers) == 1
  assert len(autoencoder.decoding_layers) == 1

  assert autoencoder.encoding_layers[0].weight.size(0) == 200
  assert autoencoder.decoding_layers[0].weight.size(1) == 200

  batch_size = 32
  input = torch.rand(batch_size, 5)
  input_items = torch.LongTensor([10, 126, 452, 29, 34])
  output = autoencoder(input, input_items=input_items,
                       target_items=input_items)

  assert output.size(0) == batch_size
  assert output.size(1) == input_items.size(0)

  target_items = torch.LongTensor([31, 14, 95, 49, 10, 36, 239])
  output = autoencoder(input, input_items=input_items,
                       target_items=target_items)
  assert output.size(0) == batch_size
  assert output.size(1) == target_items.size(0)

  output = autoencoder(input, input_items=input_items)
  assert output.size(0) == batch_size
  assert output.size(1) == 500
