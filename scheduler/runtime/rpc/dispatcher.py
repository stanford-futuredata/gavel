from __future__ import print_function

from multiprocessing.pool import ThreadPool
import subprocess
import time
import os

class Dispatcher:
    def __init__(self, worker_id, gpu_id, worker_rpc_client):
        self._thread_pool = ThreadPool()
        self._worker_id = worker_id
        self._gpu_id = gpu_id
        self._worker_rpc_client = worker_rpc_client

    def launch_job(self, job):
        # TODO: add error handling
        start_time = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self._gpu_id))
        command = '%s %s %d' % (job.command, job.num_steps_arg, job.total_steps)
        output = subprocess.run(command,
                                env=env,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=True).stdout.decode('utf-8')

        """
        output = subprocess.check_output(command,
                                         stderr=subprocess.STDOUT,
                                         shell=True).decode('utf-8')
        """
        print('flag4')
        execution_time = time.time() - start_time
        print("Job ID: %s, Command: '%s', "
              "Num_steps: %d, Execution time: %.3f seconds, "
              "Output:" % (job.job_id,
                           command,
                           job.total_steps,
                           execution_time), output)
        # TODO: add error handling
        self._worker_rpc_client.notify_scheduler(job.job_id,
                                                 self._worker_id,
                                                 execution_time,
                                                 job.total_steps)

    def dispatch_job(self, job):
        self._thread_pool.apply_async(self.launch_job, (job,))

    def shutdown(self):
        self._thread_pool.terminate()
