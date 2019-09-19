from __future__ import print_function

from multiprocessing.pool import ThreadPool
import subprocess
import sys
import time
import os

class Dispatcher:
    def __init__(self, worker_id, gpu_id, done_callback, timeout):
        self._thread_pool = ThreadPool()
        self._worker_id = worker_id
        self._gpu_id = gpu_id
        self._done_callback = done_callback
        self._timeout = timeout

    def launch_job(self, job):
        start_time = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self._gpu_id))
        command = '%s %s %d' % (job.command, job.num_steps_arg, job.total_steps)
        command += ' --throughput_estimation_interval %d' % (max(1, job.total_steps // 100))
        print('Running \"%s\"' % (command))

        try:
            proc = subprocess.run(command,
                                  env=env,
                                  check=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  shell=True,
                                  timeout=self._timeout)
            execution_time = time.time() - start_time
            output = proc.stdout.decode('utf-8')
            print("Job ID: %s, Command: '%s', "
                  "Num_steps: %d, Execution time: %.3f seconds, "
                  "Output:" % (str(job.job_id),
                               command,
                               job.total_steps,
                               execution_time), output)

        except subprocess.CalledProcessError as e:
            print('Job %s failed with error code %d' % (str(job.job_id),
                                                        e.returncode))
            output = ''
            if e.args is not None:
              args_output = 'Args: %s' % (str(e.args))
              print(args_output)
              output += args_output
              output += '\n'
            if e.stdout is not None:
              stdout_output = 'Stdout: %s' % (e.stdout)
              print(stdout_output)
              output += stdout_output
              output += '\n'
            if e.stderr is not None:
              stderr_output = 'Stderr: %s' % (e.stderr)
              print(stderr_output)
              output += stderr_output
            execution_time = -1
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            output = e.stdout.decode('utf-8')
            print("Job ID: %s, Command: '%s', "
                  "Num_steps: %d, Execution time: %.3f seconds, "
                  "Output:" % (str(job.job_id),
                               command,
                               job.total_steps,
                               execution_time), output)
        sys.stdout.flush()
        self._done_callback(job.job_id,
                            self._worker_id,
                            job.total_steps,
                            execution_time,
                            output)

    def dispatch_job(self, job):
        self._thread_pool.apply_async(self.launch_job, (job,))

    def shutdown(self):
        self._thread_pool.terminate()
