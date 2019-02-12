from __future__ import print_function

from multiprocessing.pool import ThreadPool
import subprocess

class Dispatcher:
    def __init__(self, worker_rpc_client):
        self._thread_pool = ThreadPool()
        self._worker_rpc_client = worker_rpc_client
  
    def launch_job(self, job):
        output = subprocess.check_output(job.command(),
                                         stderr=subprocess.STDOUT,
                                         shell=True)
        print("Job ID: %d, Command: '%s', "
              "Num_epochs: %d, Output:" % (job.job_id(),
                                           job.command(),
                                           job.num_epochs()), output)
        try:
            self._worker_rpc_client.notify_scheduler(job.job_id())
        except Exception as e:
            print(e)

    def dispatch_job(self, job):
        self._thread_pool.apply_async(self.launch_job, (job,))
