Notes about Tiresias
====================

- Tiresias tries to minimize average Job Completion Time
- Tiresias has four main insights:
  - It's hard to accurately predict how long a distributed deep learning
    training job is going to take
  - It's not sufficient to use time-based or resource-based heuristics to
    optimize for JCT
  - Preemption for distributed training jobs is somewhat expensive (since it
    involves moving parameters to/from GPU)
  - JCT included queuing delay, and a lot of queuing delay is caused by
    unneeded consolidation (a distributed training job running on a multi-GPU
    server to leverage faster interconnects)
- Approach:
  - Give each job a priority, adapt this priority with time, thus preventing the
    need to have an accurate estimate for how long the job is going to take
  - Use a 2-dimensional scheduler (2D-LAS and 2D-Gittens Index) to make scheduling
    decisions based on attained service (time executed times number of resources
    given)
  - Use discretized priorities to reduce the number of preemptions
  - Only consolidate when needed (for example, models like VGG-16 and AlexNet
    have some very big weight tensors, which are expensive to communicate over
    inter-machine communication links)
