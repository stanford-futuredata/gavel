from filelock import FileLock
import json
from multiprocessing.pool import ThreadPool
import math
import logging
import numa
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import utils

CUDA_MPS_PIPE_DIRECTORY = '/tmp/nvidia-mps'
CUDA_MPS_LOG_DIRECTORY = '/tmp/nvidia-log'
MAX_CPUS_PER_GPU = 8
LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class Dispatcher:
    def __init__(self, round_duration, gpu_ids, worker_rpc_client,
                 sched_addr, sched_port, run_dir, data_dir, checkpoint_dir,
                 use_mps=False):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._thread_pool = ThreadPool()
        self._round_duration = round_duration
        self._worker_rpc_client = worker_rpc_client
        self._sched_addr = sched_addr
        self._sched_port = sched_port
        self._run_dir = run_dir
        self._data_dir = data_dir
        self._checkpoint_dir = checkpoint_dir
        self._gpu_ids = gpu_ids
        self._gpu_queue = queue.Queue(len(self._gpu_ids))
        self._job_assignments = {}
        self._lock = threading.Lock()
        for gpu_id in self._gpu_ids:
            self._gpu_queue.put(gpu_id)
        self._configure_numa()
        self._use_mps = use_mps
        if use_mps:
            self._mps_initially_enabled = self._mps_status()
            if self._mps_initially_enabled:
                self._logger.info('CUDA MPS already running')
            else:
                mps_enabled = self._enable_mps()
                if not mps_enabled:
                    raise RuntimeError('Failed to enable CUDA MPS')

    def _configure_numa(self):
        self._numa_available = numa.available()
        if not self._numa_available:
            return
        num_numa_nodes = numa.get_max_node() + 1
        self._numa_cpu_map = {}
        num_gpus = len(self._gpu_ids)

        # Calculate how many CPUs to allocate for each GPU. Ensure this number
        # is a power of 2.
        num_cpus = 0
        for i in range(num_numa_nodes):
            num_cpus += len(numa.node_to_cpus(i))
        num_cpus_per_gpu = min(MAX_CPUS_PER_GPU, max(num_cpus // num_gpus, 1))
        num_cpus_per_gpu = pow(2, round(math.log(num_cpus_per_gpu, 2)))

        # Find blocks of contiguous CPUs.
        contiguous_blocks = []
        for i in range(num_numa_nodes):
            cpus = sorted(numa.node_to_cpus(i))
            contiguous_block = [cpus[0]]
            for j in range(1, len(cpus)):
                if (cpus[j] - cpus[j-1] == 1 and
                    len(contiguous_block) < num_cpus_per_gpu):
                    contiguous_block.append(cpus[j])
                else:
                    contiguous_blocks.append((contiguous_block,
                                              len(contiguous_block)))
                    contiguous_block = [cpus[j]]
            if len(contiguous_block) > 0:
                contiguous_blocks.append((contiguous_block,
                                          len(contiguous_block)))
        contiguous_blocks.sort(key=lambda x: x[-1], reverse=True)

        # Assign CPUs to GPUs.
        block_idx = 0
        for i in range(num_gpus):
            self._numa_cpu_map[i] = []
            while len(self._numa_cpu_map[i]) < num_cpus_per_gpu:
                self._numa_cpu_map[i] += contiguous_blocks[block_idx][0]
                block_idx = (block_idx + 1) % len(contiguous_blocks)
            self._logger.info('GPU {gpu} assigned CPUs {cpus}'.format
                    (gpu=i, cpus=str(self._numa_cpu_map[i])))

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
            self._logger.info('Successfully enabled CUDA MPS')
            return True
        except subprocess.CalledProcessError as e:
            error_message = 'Unable to start CUDA MPS:\n'
            if e.stdout is not None:
                error_message += 'Stdout:\n%s' % (e.stdout.decode('utf-8'))
            if e.stderr is not None:
                error_message += 'Stderr:\n%s' % (e.stderr.decode('utf-8'))
            self._logger.error(error_message)
        return False

    def _shutdown_mps(self):
        command = 'echo quit | nvidia-cuda-mps-control'
        subprocess.run(command, shell=True)
        self._logger.info('Shut down CUDA MPS')

    def _construct_command(self, job, gpu_id, worker_id):
        checkpoint_dir = os.path.join(self._checkpoint_dir,
                                      'job_id=%d' % (job.job_id))
        with self._lock:
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)

        if job.needs_data_dir:
            command = job.command % (self._data_dir)
        else:
            command = job.command

        command = '%s --local_rank %d' % (command, gpu_id)
        command = '%s %s %d' % (command, job.num_steps_arg, job.total_steps)
        command = '%s --checkpoint_dir %s' % (command, checkpoint_dir)

        if self._numa_available:
            cpus = self._numa_cpu_map[gpu_id]
            cpus_str = ','.join([str(cpu) for cpu in cpus])
            command = 'numactl --physcpubind=%s %s' % (cpus_str, command)

        return command

    def _prepare_gavel_info(self, gpu_id, job_id, worker_id):
        # NOTE: We assume that two packed jobs will not share the same
        # checkpoint directory. This is currently enforced by the dispatcher,
        # but in the future the dispatcher may no longer be responsible for
        # creating checkpoint directories.
        checkpoint_dir = \
            os.path.join(self._checkpoint_dir, 'job_id=%d' % (job_id))
        lock_file = os.path.join(checkpoint_dir, '.gavel.lock')
        gavel_lock = FileLock(lock_file)
        gavel_file = os.path.join(checkpoint_dir, '.gavel.json')
        with gavel_lock:
            if os.path.exists(gavel_file):
                    with open(gavel_file, 'r') as f:
                        gavel_info = json.load(f)
            else:
                gavel_info = {}
            if 'job_info' not in gavel_info:
                gavel_info['job_info'] = {}
            gavel_info['job_info'][gpu_id] = {
                'job_id': job_id,
                'worker_id': worker_id,
                'steps': 0,
                'duration': 0,
            }
            if 'server_info' not in gavel_info:
                gavel_info['server_info'] = {
                    'addr': self._sched_addr,
                    'port': self._sched_port,
                }
            with open(gavel_file, 'w') as f:
                json.dump(gavel_info, f)

        self._logger.debug(
            'Gavel info for job {job_id}, worker {worker_id} (GPU {gpu_id}):\n'
            '{output}'.format(job_id=job_id, worker_id=worker_id,
                              gpu_id=gpu_id, output=json.dumps(gavel_info)))

    def _get_steps_and_execution_time(self, gpu_id, job_id):
        gpu_id = str(gpu_id)
        checkpoint_dir = os.path.join(self._checkpoint_dir,
                                      'job_id=%d' % (job_id))
        lock_file = os.path.join(checkpoint_dir, '.gavel.lock')
        gavel_lock = FileLock(lock_file)
        gavel_file = os.path.join(checkpoint_dir, '.gavel.json')
        with gavel_lock:
            try:
                with open(gavel_file, 'r') as f:
                    gavel_info = json.load(f)
                steps = int(gavel_info['job_info'][gpu_id]['steps'])
                execution_time = \
                    float(gavel_info['job_info'][str(gpu_id)]['duration'])
                del gavel_info['job_info'][gpu_id]
                with open(gavel_file, 'w') as f:
                    json.dump(gavel_info, f)
            except Exception as e:
                self._logger.error(
                    'Could not read steps and execution time: {0}'.format(e))
                return (0, 0)
        return steps, execution_time

    def _kill_job(self, pid):
        self._logger.debug('Killing process {0}'.format(pid))
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError as e:
            self._logger.debug('Could not find process {0}'.format(pid))
        except Exception as e:
            self._logger.error(
                'Could not kill process {0}: {1}'.format(pid, e))

    def _kill_jobs(self, job_id=None):
        with self._lock:
            if job_id is not None:
                self._logger.debug('Killing job {0}...'.format(job_id))
            else:
                self._logger.debug('Killing all jobs!')
            if job_id is not None:
                pids = utils.get_pid_for_job(job_id)
                self._logger.debug('PIDs for job {0}: {1}'.format(
                    job_id, str(pids)))
                for pid in pids:
                    self._kill_job(pid)
            else:
                for job_id in self._job_assignments:
                    pids = utils.get_pid_for_job(job_id)
                    for pid in pids:
                        self._kill_job(pid)
            self._logger.debug('Finished killing job(s)')

    def launch_job(self, job, command, worker_id, gpu_id):
        output = ''
        cwd = os.path.join(self._run_dir, job.working_directory)
        try:
            proc = subprocess.run(command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=cwd,
                                  shell=True)
            output = proc.stdout.decode('utf-8').strip()
            completed_steps, execution_time = \
                self._get_steps_and_execution_time(gpu_id, job.job_id)
            if completed_steps is None:
                self._logger.error('Could not get completed steps for job '
                                   '{job_id} (worker {worker_id}), '
                                   'Output:\n{output}'.format(
                                       job_id=job.job_id, worker_id=worker_id,
                                       output=output))
                completed_steps = 0
                self._kill_jobs(job_id=job.job_id)
            else:
                self._logger.info(
                    'Job ID: {job_id}, '
                    'Worker ID: {worker_id}, '
                    'Num steps: {num_steps}, '
                    'Execution time: {execution_time:.2f} seconds, '
                    'Output:\n{output}'.format(
                        job_id=job.job_id, worker_id=worker_id,
                        num_steps=completed_steps,
                        execution_time=execution_time, output=output))
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
            self._logger.error(error_message)
            execution_time = -1
            completed_steps = 0
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
            self._logger.error(error_message)
            execution_time = -1
            completed_steps = 0
            self._kill_jobs(job_id=job.job_id)
        except Exception as e:
            self._logger.error('Dispatcher failed: %s' % (str(e)))
            execution_time = -1
            completed_steps = 0

        return [job.job_id, execution_time, completed_steps]

    def _dispatch_jobs_helper(self, jobs, worker_id):
        job_ids = [job.job_id for job in jobs]
        self._logger.debug(
            'Requesting GPU for job(s) {0} (worker {1})...'.format(
                job_ids, worker_id))
        gpu_id = self._gpu_queue.get()
        self._logger.debug('Using GPU {gpu_id} for job(s) '
                           '{job_id} (worker {worker_id})'.format(
                               gpu_id=gpu_id, job_id=job_ids,
                               worker_id=worker_id))
        with self._lock:
            for job_id in job_ids:
                if job_id not in self._job_assignments:
                    self._job_assignments[job_id] = []
                self._job_assignments[job_id].append(gpu_id)

        self._logger.debug('Constructing commands for '
                           'worker {0}...'.format(worker_id))

        success = True
        commands = []
        for job in jobs:
            try:
                command = self._construct_command(job, gpu_id, worker_id)
                commands.append(command)
            except Exception as e:
                self._logger.error(
                    'Failed to construct command for job {0}: {1}'.format(
                        job_id, e))
                success = False
                break

        if success:
            for job in jobs:
                try:
                    self._prepare_gavel_info(gpu_id, job.job_id, worker_id)
                except Exception as e:
                    self._logger.error('Could not prepare Gavel info for '
                                        'job {0}: {1}'.format(job.job_id, e))
                    success = False
                    break

        if success:
            # Launch the jobs.
            results = []
            for job, command in zip(jobs, commands):
                self._logger.info('Running job {job_id} on GPU {gpu_id}, '
                                  'command: "{command}"'.format(
                                      job_id=job.job_id, gpu_id=gpu_id,
                                      command=command))
                results.append(self._thread_pool.apply_async(self.launch_job,
                                                             (job, command,
                                                              worker_id,
                                                              gpu_id)))
            job_descriptions = [result.get() for result in results]
        else:
            job_descriptions = [[job.job_id, -1, 0] for job in jobs]

        # Cleanup and notify the scheduler.
        self._gpu_queue.put(gpu_id)
        if len(jobs) == 1:
            self._logger.debug('Job {0} has completed, '
                               'notifying scheduler...'.format(jobs[0].job_id))
        else:
            job_ids = [job.job_id for job in jobs]
            self._logger.debug('Jobs {0} have completed, '
                               'notifying scheduler...'.format(job_ids))
        self._worker_rpc_client.notify_scheduler(worker_id,
                                                 job_descriptions)

    def dispatch_jobs(self, jobs, worker_id):
        self._thread_pool.apply_async(self._dispatch_jobs_helper,
                                      (jobs, worker_id,))

    def reset(self):
        self._logger.debug('Resetting dispatcher...')
        self._kill_jobs()
        self._job_assignments = {}
        self._thread_pool = ThreadPool()
        self._gpu_queue = queue.Queue(len(self._gpu_ids))
        for gpu_id in self._gpu_ids:
            self._gpu_queue.put(gpu_id)
        self._logger.debug('Finished resetting dispatcher')

    def shutdown(self, shut_down_mps=True):
        self._logger.debug('Shutting down dispatcher...')
        self._kill_jobs()
        self._thread_pool.terminate()
        if self._use_mps and shut_down_mps and not self._mps_initially_enabled:
            self._shutdown_mps()
        self._logger.debug('Finished shutting down dispatcher')
