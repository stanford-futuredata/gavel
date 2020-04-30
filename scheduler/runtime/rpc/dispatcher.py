from __future__ import print_function

from multiprocessing.pool import ThreadPool
import queue
import signal
import subprocess
import sys
import threading
import time
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import utils

BUFFER_TIME = 180
CUDA_MPS_PIPE_DIRECTORY = '/tmp/nvidia-mps'
CUDA_MPS_LOG_DIRECTORY = '/tmp/nvidia-log'

class Dispatcher:
    def __init__(self, round_duration, gpu_ids, worker_rpc_client,
                 sched_addr, sched_port, run_dir, checkpoint_dir,
                 write_queue, use_mps=False):
        self._thread_pool = ThreadPool()
        self._round_duration = round_duration
        self._worker_rpc_client = worker_rpc_client
        self._sched_addr = sched_addr
        self._sched_port = sched_port
        self._run_dir = run_dir
        self._checkpoint_dir = checkpoint_dir
        self._gpu_ids = gpu_ids
        self._gpu_queue = queue.Queue(len(self._gpu_ids))
        self._write_queue = write_queue
        self._job_assignments = {}
        self._lock = threading.Lock()
        for gpu_id in self._gpu_ids:
            self._gpu_queue.put(gpu_id)
        self._use_mps = use_mps
        if use_mps:
            self._mps_initially_enabled = self._mps_status()
            if self._mps_initially_enabled:
                self._write_queue.put('CUDA MPS already running')
            else:
                mps_enabled = self._enable_mps()
                if not mps_enabled:
                    raise RuntimeError('Failed to enable CUDA MPS')

    def _mps_status(self):
        """Returns True if MPS is running."""
        command = 'ps -ef | grep mps'
        output = subprocess.run(command, stdout=subprocess.PIPE, check=True,
                                shell=True).stdout.decode('utf-8').strip()
        running = False
        for line in output.split('\n'):
            if 'nvidia-cuda-mps-control -d' in line:
                running = True
                break
        return running

    def _enable_mps(self):
        # Set logging environment variables.
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = CUDA_MPS_PIPE_DIRECTORY
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = CUDA_MPS_LOG_DIRECTORY

        # Start the daemon.
        command = 'nvidia-cuda-mps-control -d'

        try:
            output =\
                subprocess.run(command, stdout=subprocess.PIPE,
                               check=True,
                               shell=True).stdout.decode('utf-8').strip()
            self._write_queue.put('Successfully enabled CUDA MPS')
            return True
        except subprocess.CalledProcessError as e:
            error_message = 'Unable to start CUDA MPS:\n'
            if e.stdout is not None:
                error_message += 'Stdout:\n%s' % (e.stdout.decode('utf-8'))
            if e.stderr is not None:
                error_message += 'Stderr:\n%s' % (e.stderr.decode('utf-8'))
            self._write_queue.put(error_message)
        return False

    def _shutdown_mps(self):
        command = 'echo quit | nvidia-cuda-mps-control'
        subprocess.run(command, shell=True)
        self._write_queue.put('Shut down CUDA MPS')

    def _construct_command(self, job, gpu_id, worker_id):
        checkpoint_dir = os.path.join(self._checkpoint_dir,
                                      'job_id=%d' % (job.job_id))
        with self._lock:
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)

        if job.needs_data_dir:
            command = job.command % (self._run_dir, self._run_dir)
        else:
            command = job.command % (self._run_dir)

        command = '%s --local_rank %d' % (command, gpu_id)
        command = '%s %s %d' % (command, job.num_steps_arg, job.total_steps)
        command = '%s --checkpoint_dir %s' % (command, checkpoint_dir)
        command = ('%s --job_id %d --worker_id %d --sched_addr %s '
                   '--sched_port %d' % (command, job.job_id, worker_id,
                                        self._sched_addr, self._sched_port))
        return command

    def _get_steps_from_output(self, output):
        steps = None
        for line in output.split('\n'):
            if line.startswith('[GavelIterator]'):
                steps = int(line.split('[GavelIterator]')[-1])
        return steps

    def _kill_job(self, pid, gpu_id):
        self._write_queue.put(
            'Killing process %d on GPU %d' % (pid, gpu_id))
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError as e:
            self._write_queue.put(
                'Could not find process %d on GPU %d' % (pid, gpu_id))
        except Exception as e:
            self._write_queue.put(
                'Could not kill process %d on GPU %d: %s' % (pid, gpu_id,
                                                             str(e)))


    def _kill_jobs(self, job_id=None):
        with self._lock:
            gpu_processes = utils.get_gpu_processes()
            if job_id is not None:
                # Kill all jobs.
                for gpu_id in self._job_assignments[job_id]:
                    if gpu_id not in gpu_processes:
                        continue
                    for pid in gpu_processes[gpu_id]:
                        self._kill_job(pid, gpu_id)
            else:
                for gpu_id in gpu_processes:
                    for pid in gpu_processes[gpu_id]:
                        self._kill_job(pid, gpu_id)

    def launch_job(self, job, command, worker_id):
        start_time = time.time()
        output = ''
        try:
            proc = subprocess.run(command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  timeout=(self._round_duration + BUFFER_TIME),
                                  shell=True)
            execution_time = time.time() - start_time
            output = proc.stdout.decode('utf-8').strip()
            completed_steps = self._get_steps_from_output(output)
            if completed_steps is None:
                self._write_queue.put('Could not get completed steps for job '
                                      '%s (worker %d), '
                                      'Output:\n%s' % (str(job.job_id),
                                                       worker_id,
                                                       output))
                completed_steps = 0
                self._kill_jobs(job_id=job.job_id)
            else:
                self._write_queue.put('Job ID: %s, '
                                      'Worker ID: %d '
                                      'Num steps: %d, '
                                      'Execution time: %.3f seconds, '
                                      'Output:\n%s' % (str(job.job_id),
                                                       worker_id,
                                                       completed_steps,
                                                       execution_time,
                                                       output))
        except subprocess.CalledProcessError as e:
            error_message = ('Job %s (worker %d) failed with '
                             'error code %s' % (str(job.job_id),
                                                worker_id,
                                                str(e.returncode)))
            if e.args is not None:
                error_message += '\nArgs: %s' % (str(e.args))
            if e.stdout is not None:
                error_message += '\nStdout: %s' % (e.stdout.decode('utf-8'))
            if e.stderr is not None:
                error_message += '\nStderr: %s' % (e.stderr.decode('utf-8'))
            self._write_queue.put(error_message)
            execution_time = -1
            completed_steps = 0
            output = ''
            self._kill_jobs(job_id=job.job_id)
        except subprocess.TimeoutExpired as e:
            error_message = 'Job %s (worker %d) timed out' % (str(job.job_id),
                                                              worker_id)
            if e.args is not None:
                error_message += '\nArgs: %s' % (str(e.args))
            if e.stdout is not None:
                error_message += '\nStdout: %s' % (e.stdout.decode('utf-8'))
            if e.stderr is not None:
                error_message += '\nStderr: %s' % (e.stderr.decode('utf-8'))
            self._write_queue.put(error_message)
            execution_time = -1
            completed_steps = 0
            self._kill_jobs(job_id=job.job_id)
        except Exception as e:
            self._write_queue.put('Dispatcher failed: %s' % (str(e)))
            execution_time = -1
            completed_steps = 0

        return [job.job_id, execution_time, completed_steps]

    def _dispatch_jobs_helper(self, jobs, worker_id):
        job_ids = [job.job_id for job in jobs]
        self._write_queue.put(
            'Requesting GPU for job(s) %s (worker %d)...' % (str(job_ids),
                                                             worker_id))
        gpu_id = self._gpu_queue.get()
        self._write_queue.put('Using GPU %d for job(s) '
                              '%s (worker %d)' % (gpu_id, str(job_ids),
                                                  worker_id))
        with self._lock:
            for job_id in job_ids:
                if job_id not in self._job_assignments:
                    self._job_assignments[job_id] = []
                self._job_assignments[job_id].append(gpu_id)

        self._write_queue.put('Constructing commands for '
                              'worker %d...' % (worker_id))

        commands = \
            [self._construct_command(job, gpu_id, worker_id) for job in jobs]
        results = []

        # Launch the jobs.
        for job, command in zip(jobs, commands):
            self._write_queue.put('Running job %d on GPU %d, '
                                  'command: "%s"' % (job.job_id, gpu_id,
                                                     command))
            results.append(self._thread_pool.apply_async(self.launch_job,
                                                         (job, command,
                                                          worker_id)))
        job_descriptions = [result.get() for result in results]

        # Cleanup and notify the scheduler.
        self._gpu_queue.put(gpu_id)
        if len(jobs) == 1:
            self._write_queue.put('Job %d has completed, '
                                  'notifying scheduler...' % (jobs[0].job_id))
        else:
            job_ids = [job.job_id for job in jobs]
            self._write_queue.put('Jobs %s have completed, '
                                  'notifying scheduler...' % (str(job_ids)))
        self._worker_rpc_client.notify_scheduler(worker_id,
                                                 job_descriptions)

    def dispatch_jobs(self, jobs, worker_id):
        self._thread_pool.apply_async(self._dispatch_jobs_helper,
                                      (jobs, worker_id,))

    def reset(self):
        self._write_queue.put('Resetting dispatcher')
        self._kill_jobs()
        self.shutdown(shut_down_mps=False)
        self._job_assignments = {}
        self._thread_pool = ThreadPool()
        self._gpu_queue = queue.Queue(len(self._gpu_ids))
        for gpu_id in self._gpu_ids:
            self._gpu_queue.put(gpu_id)

    def shutdown(self, shut_down_mps=True):
        self._thread_pool.terminate()
        self._thread_pool.join()
        if self._use_mps and shut_down_mps and not self._mps_initially_enabled:
            self._shutdown_mps()
