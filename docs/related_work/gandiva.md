Notes about Gandiva
===================

Scheduler:
----------
Scheduler is based on greedy heuristics. The scheduler first tries to place
jobs on servers with the same affinity (number of GPUs used by jobs on the
same server; for example, a 8-GPU server which uses 4 GPUs each would have
an affinity of 4). Among available servers with the same affinity, less
loaded servers are picked. If no such server exists, but another server
exists with no assigned affinity _and_ enough available GPUs, this server
is picked (and assigned this new affinity). Once these two fail, the scheduler
falls back on servers with the wrong affinity, but available GPUs (such
fragmentation can be cleaned up later using job migration); otherwise, the
scheduler resorts to oversubscription on servers with the same affinity.
If this all fails, the job is added to the queue, to be scheduled at a future
point when resource are available.
[see Algorithm 1 in paper for an implementation of this]

When over-subscribed, Gandiva uses stop-and-restart to ensure fairness within
a server.


Packing:
--------

"Packing is considered only during overload. The basic idea behind packing is
to run two or more jobs simultaneously on a GPU to increase efficiency. If
the memory requirements of the packing jobs combined are higher than GPU
memory, the overhead of “paging” from CPU memory is significantly high [16]
that packing is not effective."

"Analytically modeling performance of packing is a challenging problem given
the heterogeneity of DLT jobs. Instead, Gandiva relies on a greedy heuristic
to pack jobs. When jobs arrive, we always run them in exclusive mode using
suspend-resume and collect profiling information (GPU utilization, memory and
job progress rate). Based on the profiling data, the scheduler maintains a list
of jobs sorted by their GPU utilization. The scheduler greedily picks the job
with the lowest GPU utilization and attempts to pack it on a GPU with the
lowest GPU utilization. We only do this when the combined memory utilization
of the packed jobs do not exceed the overall memory of the GPU. Packing is
deemed successful when the total throughput of packed jobs is greater than
time-slicing. If packing is unsuccessful, we undo the packing and try the next
lowest utilization GPU. If the packing is successful, we find the next lower
utilization job and repeat this process. Based on our evaluation, we find that
this simple greedy heuristic achieves 26% efficiency gains."
