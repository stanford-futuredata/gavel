import numpy as np

import recoder.utils as utils
from recoder.data import RecommendationDataLoader

from multiprocessing import Process, Queue


def average_precision(x, y, k, normalize=True):
  x = x[:k]
  x_in_y = np.isin(x, y, assume_unique=True).astype(np.int)

  tp = x_in_y.cumsum()  # true positives at every position in x_in_y
  precision = tp / (1 + np.arange(len(x)))  # precision at every position
  precision_drecall = np.multiply(precision, x_in_y)  # precision * delta_recall at every position

  normalization = min(k, len(y)) if normalize else len(y)
  ap = precision_drecall.sum() / normalization

  return ap


def recall(x, y, k, normalize=True):
  x = x[:k]
  x_in_y = np.isin(x, y, assume_unique=True).astype(np.int)
  normalization = min(k, len(y)) if normalize else len(y)
  _recall = x_in_y.sum() / normalization

  return _recall


def dcg(x, y, k):
  x = x[:k]
  x_in_y = np.isin(x, y, assume_unique=True).astype(np.int)
  cg = x_in_y / np.log2(2 + np.arange(len(x)))  # cumulative gain at every position
  _dcg = cg.sum()

  return _dcg


def ndcg(x, y, k):
  dcg_k = dcg(x, y, k)
  idcg_k = dcg(y, y, k)
  ndcg_k = dcg_k / idcg_k
  return ndcg_k


class Metric(object):
  """
  A Base class for metrics. All metrics should implement the ``evaluate`` function.

  Args:
    metric_name (str): metric name. useful for representing it as string
      (printing) and hashing.
  """
  def __init__(self, metric_name):
    self.metric_name = metric_name

  def __str__(self):
    return self.metric_name

  def __hash__(self):
    return self.metric_name.__hash__()

  def evaluate(self, x, y):
    """
    Evaluates the recommendations with respect to the items the user interacted with.

    Args:
      x (list): items recommended for the user
      y (list): items the user interacted with

    Returns:
      float: metric value
    """
    raise NotImplementedError


class AveragePrecision(Metric):
  """
  Computes the average precision @ K of the recommended items.

  Args:
    k (int): the cut position of the recommended list
    normalize (bool, optional): if True, normalize the value to 1 (divide by k)
      if k is less than the number of items the user interacted with, otherwise normalize
      only by number of items the user interacted with.
  """

  def __init__(self, k, normalize=True):
    super().__init__(metric_name='AveragePrecision@{}'.format(k))
    self.k = k
    self.normalize = normalize

  def evaluate(self, x, y):
    return average_precision(x, y, k=self.k, normalize=self.normalize)


class Recall(Metric):
  """
  Computes the recall @ K of the recommended items.

  Args:
    k (int): the cut position of the recommended list
    normalize (bool, optional): if True, normalize the value to 1 (divide by k)
      if k is less than the number of items in the user interactions, otherwise normalize
      only by number of items in the user interactions.
  """

  def __init__(self, k, normalize=True):
    super().__init__(metric_name='Recall@{}'.format(k))
    self.k = k
    self.normalize = normalize

  def evaluate(self, x, y):
    return recall(x, y, k=self.k, normalize=self.normalize)


class NDCG(Metric):
  """
  Computes the normalized discounted cumulative gain @ K of the recommended items.

  Args:
    k (int): the cut position of the recommended list
  """

  def __init__(self, k):
    super().__init__(metric_name='NDCG@{}'.format(k))
    self.k = k

  def evaluate(self, x, y):
    return ndcg(x, y, k=self.k)


class RecommenderEvaluator(object):
  """
  Evaluates a :class:`recoder.recommender.Recommender` given a set of :class:`Metric`

  Args:
    recommender (Recommender): the Recommender to evaluate
    metrics (list): list of metrics used to evaluate the recommender
  """

  def __init__(self, recommender, metrics):
    self.recommender = recommender
    self.metrics = metrics

  def evaluate(self, eval_dataset, batch_size=1,
               num_users=None, num_workers=0):
    """
    Evaluates the recommender with an evaluation dataset.

    Args:
      eval_dataset (RecommendationDataset): the dataset to use
        in evaluating the model
      batch_size (int): the size of the users batch passed to the recommender
      num_users (int, optional): the number of users from the dataset to evaluate on. If None,
        evaluate on all users
      num_workers (int, optional): the number of workers to use on evaluating
        the recommended items. This is useful if the recommender runs on GPU, so the
        evaluation can run in parallel.

    Returns:
      dict: A dict mapping each metric to the list of the metric values on each
      user in the dataset.
    """
    dataloader = RecommendationDataLoader(eval_dataset, batch_size=batch_size,
                                          collate_fn=lambda _: _)

    results = {}
    for metric in self.metrics:
      results[metric] = []

    if num_workers > 0:
      input_queue = Queue()
      results_queues = [Queue() for _ in range(num_workers)]

      def evaluate(input_queue, results_queue, metrics):
        results = {}
        for metric in self.metrics:
          results[metric.metric_name] = []

        while True:
          x, y = input_queue.get(block=True)

          if x is None:
            break

          for metric in metrics:
            results[metric.metric_name].append(metric.evaluate(x, y))

        results_queue.put(results)

      workers = [Process(target=evaluate, args=(input_queue, results_queues[p_idx], self.metrics))
                 for p_idx in range(num_workers)]

      for worker in workers:
        worker.start()

    processed_num_users = 0
    for batch in dataloader:
      input, target = batch

      recommendations = self.recommender.recommend(input)

      relevant_items = [target.interactions_matrix[i].nonzero()[1] for i in range(len(target.users))]

      for x, y in zip(recommendations, relevant_items):
        if num_workers > 0:
          input_queue.put((x, y))
        else:
          for metric in self.metrics:
            results[metric].append(metric.evaluate(x, y))

      processed_num_users += len(target.users)
      if num_users is not None and processed_num_users >= num_users:
        break

    for _ in range(num_workers):
      input_queue.put((None, None))

    if num_workers > 0:

      for results_queue in results_queues:
        queue_results = results_queue.get()
        for metric in self.metrics:
          results[metric].extend(queue_results[metric.metric_name])

      for worker in workers:
        worker.join()

    return results
