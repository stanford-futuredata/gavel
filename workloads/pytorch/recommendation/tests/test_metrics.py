from recoder.metrics import AveragePrecision, Recall, NDCG, ndcg, dcg

import pytest

import numpy as np


RTOL = 1e-9
ATOL = 0.0


test_ap_tests = [
  (np.arange(10), [0, 2, 5, 8, 9], 10, False, 1 / 5 * (1 + 2 / 3 + 3 / 6 + 4 / 9 + 5 / 10)),
  (np.arange(10), [1, 4, 5, 6, 12], 10, False, 1 / 5 * (1 / 2 + 2 / 5 + 3 / 6 + 4 / 7 + 0)),
  (np.arange(10), [0, 1, 2, 3, 4], 10, False, 1),
  (np.arange(10), [0, 2, 5, 8, 9], 3, True, 1 / 3 * (1 + 2 / 3)),
  (np.arange(10), [1, 4, 5, 6, 12], 3, True, 1 / 3 * (1 / 2)),
]
@pytest.mark.parametrize("x, y, k, normalize, expected_value",
                         test_ap_tests)
def test_ap(x, y, k, normalize, expected_value):
  metric = AveragePrecision(k=k, normalize=normalize)

  assert np.isclose(metric.evaluate(x, y), expected_value, rtol=RTOL, atol=ATOL)


test_recall_tests = [
  (np.arange(10), [0, 2, 5, 8, 9], 10, False, 1),
  (np.arange(10), [1, 4, 5, 6, 12], 10, False, 4 / 5),
  (np.arange(10), [0, 2, 5, 8, 9], 3, False, 2 / 5),
  (np.arange(10), [1, 4, 5, 6, 12], 3, False, 1 / 5),
  (np.arange(10), [0, 2, 5, 8, 9], 3, True, 2 / 3),
  (np.arange(10), [1, 4, 5, 6, 12], 3, True, 1 / 3),
]
@pytest.mark.parametrize("x, y, k, normalize, expected_value",
                         test_recall_tests)
def test_recall(x, y, k, normalize, expected_value):
  metric = Recall(k=k, normalize=normalize)

  assert np.isclose(metric.evaluate(x, y), expected_value, rtol=RTOL, atol=ATOL)


test_recall_tests = [
  (np.arange(10), [0, 2, 5, 8, 9], 10, 0.8296882915641869),
  (np.arange(10), [1, 4, 5, 6, 12], 10, 0.5790560467042355),
  (np.arange(10), [0, 2, 5, 8, 9], 3, 0.7039180890341347),
  (np.arange(10), [1, 4, 5, 6, 12], 3, 0.2960819109658652),
]
@pytest.mark.parametrize("x, y, k, expected_value",
                         test_recall_tests)
def test_ndcg(x, y, k, expected_value):
  metric = NDCG(k=k)

  assert np.isclose(metric.evaluate(x, y), expected_value, rtol=RTOL, atol=ATOL)
