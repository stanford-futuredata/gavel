from recoder.embedding import AnnoyEmbeddingsIndex
from recoder.model import Recoder
import torch

def build_embeddings_all_layers(model):
  num_layers = len(model.autoencoder.layer_sizes)
  weights_v = list(model.autoencoder.parameters())[:num_layers-1]
  weights = [w.data for w in weights_v]
  weights.reverse()

  embeddings = weights[0]
  for w in weights[1:]:
    embeddings = torch.matmul(embeddings, w)

  return embeddings.numpy()

def build_embeddings_first_layer(model):
  return model.autoencoder.en_embedding_layer.weight.data

model_file = 'models/ml-20m/bce_ns_d_0.0_n_0.5_200_epoch_100.model'
model = Recoder()
model.init_from_model_file(model_file)

index = AnnoyEmbeddingsIndex(embeddings=build_embeddings_first_layer(model),
                             index_file=model_file+'.index', id_map=model.item_id_map)

index.build()
