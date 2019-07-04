# Principled GPU Cluster Scheduling

`scheduler/` contains most of the scheduling code, including the scheduling mechanism
(`scheduler.py`), implementations of performance-aware policies (`policies.py`), and
a basic communication stack between servers and clients (`runtime/`).

`workloads/` contains implementations of target workloads in PyTorch.

`docs/` contains some documentation, including writeups on some of the policies implemented,
the scheduling mechanism used, and related work.
