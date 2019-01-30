import numpy as np

import sys; sys.path.append(".")
import scheduler

class TestPolicy:
    def get_allocation(self, throughputs):
        (m, n) = throughputs.shape
        return np.full((m, n), 1.0 / n)

class TestStub:
    def run_application(self, command, job_id, worker_id,
                        num_epochs):
        print("Running application_%d on %s for %d epochs: %s" %
            (job_id, worker_id, num_epochs, command))

def get_num_epochs_to_run(job_id, worker_id):
    return 1

def test():
    worker_ids = ["v100"]
    num_applications = 10
    run_so_far = {}
    s = scheduler.Scheduler(worker_ids, TestPolicy(), TestStub(),
                            get_num_epochs_to_run)
    for j in range(num_applications):
        s.add_new_job({worker_id: 10 for worker_id in worker_ids},
                      "cmd%d" % j)
    for i in range(100):
        job_id, worker_id, num_epochs = s._schedule()
        s._schedule_callback(job_id, worker_id, num_epochs)
        s._add_available_worker_id(worker_id)
        if job_id not in run_so_far:
            run_so_far[job_id] = 0
        run_so_far[job_id] += num_epochs
        if i > 70 and job_id > 1 and job_id < 10:
            s.remove_old_job(job_id)
        if i == 90:
            s.add_new_job({worker_id: 10 for worker_id in worker_ids},
                          "cmd10")
    print()
    print("Number of epochs run:", run_so_far)


if __name__ == '__main__':
    test()
