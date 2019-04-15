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
        command = '%s %s %d' % (job.command, job.num_steps_arg, job.num_steps)
        output = subprocess.check_output(command,
                                         stderr=subprocess.STDOUT,
                                         shell=True).strip()
        execution_time = time.time() - start_time
        print("Job ID: %d, Command: '%s', "
              "Num_steps: %d, Execution time: %.3f seconds, "
              "Output:" % (job.job_id(),
                           command,
                           job.num_steps(),
                           execution_time), output)
        # TODO: add error handling
        self._worker_rpc_client.notify_scheduler(job.job_id(),
                                                 self._worker_id,
                                                 execution_time)

    def dispatch_job(self, job):
        self._thread_pool.apply_async(self.launch_job, (job,))

    def shutdown(self):
        self._thread_pool.terminate()
