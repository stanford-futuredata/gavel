from __future__ import print_function

from multiprocessing.pool import ThreadPool
import subprocess
import sys
import time
import os

class Dispatcher:
    def __init__(self, worker_id, gpu_id, worker_rpc_client):
        self._thread_pool = ThreadPool()
        self._worker_id = worker_id
        self._gpu_id = gpu_id
        self._worker_rpc_client = worker_rpc_client

    def launch_job(self, job):
        start_time = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self._gpu_id))
        command = '%s %s %d' % (job.command, job.num_steps_arg,
                                job.total_steps)
        print('Running \"%s\"' % (command))
        try:
            proc = subprocess.run(command,
                                  env=env,
                                  check=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  shell=True)
            execution_time = time.time() - start_time
            print("Job ID: %s, Command: '%s', "
                  "Num_steps: %d, Execution time: %.3f seconds, "
                  "Output:" % (str(job.job_id),
                               command,
                               job.total_steps,
                               execution_time), proc.stdout.decode('utf-8'))

        except subprocess.CalledProcessError as e:
            print('Job %s failed with error code %d' % (str(job.job_id),
                                                        e.returncode))
            if e.args is not None:
              print('Args: %s' % (str(e.args)))
            if e.stdout is not None:
              print('Stdout: %s' % (e.stdout))
            if e.stderr is not None:
              print('Stderr: %s' % (e.stderr))
            execution_time = -1
        except Exception as e:
            print('Dispatcher failed: %s' % (e))
            execution_time = -1


        sys.stdout.flush()
        self._worker_rpc_client.notify_scheduler(job.job_id,
                                                 self._worker_id,
                                                 execution_time,
                                                 job.total_steps)

    def dispatch_job(self, job):
        self._thread_pool.apply_async(self.launch_job, (job,))

    def shutdown(self):
        self._thread_pool.terminate()
