from __future__ import print_function

from multiprocessing.pool import ThreadPool
import subprocess
import time

class Dispatcher:
    def __init__(self, worker_id, worker_rpc_client):
        self._thread_pool = ThreadPool()
        self._worker_id = worker_id
        self._worker_rpc_client = worker_rpc_client

    def launch_job(self, job):
        # TODO: add error handling
        start_time = time.time()
        output = subprocess.check_output(job.command(),
                                         stderr=subprocess.STDOUT,
                                         shell=True).strip()
        execution_time = time.time() - start_time
        print("Job ID: %d, Command: '%s', "
              "Num_epochs: %d, Execution time: %.3f seconds, "
              "Output:" % (job.job_id(),
                           job.command(),
                           job.num_epochs(),
                           execution_time), output)
        # TODO: add error handling
        self._worker_rpc_client.notify_scheduler(job.job_id(),
                                                 self._worker_id,
                                                 execution_time)

    def dispatch_job(self, job):
        self._thread_pool.apply_async(self.launch_job, (job,))
