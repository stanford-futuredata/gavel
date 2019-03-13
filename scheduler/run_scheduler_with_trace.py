import argparse
import cvxpy as cp
import grpc
import numpy as np
import time

import job
import runtime.rpc.scheduler_client as scheduler_client
import scheduler

class Policy:
    def flatten(self, d):
        """Converts a 2-level dict to a NumPy array."""

        job_ids = list(d.keys())
        if len(job_ids) == 0:
            return None, None
        worker_types = list(d[job_ids[0]].keys())
        if len(worker_types) == 0:
            return None, None
        m = []
        for job_id in job_ids:
            m_row = []
            for worker_type in worker_types:
                m_row.append(d[job_id][worker_type])
            m.append(m_row)
        return np.array(m), (job_ids, worker_types)

    def unflatten(self, m, index):
        """Converts a NumPy array to a 2-level dict."""

        (job_ids, worker_types) = index
        d = {}
        for i in range(len(job_ids)):
            d[job_ids[i]] = {}
            for j in range(len(worker_types)):
                d[job_ids[i]][worker_types[j]] = m[i][j]
        return d


class IsolatedPolicy(Policy):
    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        return super().unflatten(np.full((m, n), 1.0 / (m * n)), index)


class KSPolicy(Policy):
    def get_allocation_flattened(self, throughputs):
        x = cp.Variable(throughputs.shape)
        objective = cp.Maximize(cp.min(cp.sum(cp.multiply(throughputs, x), axis=1)))
        constraints = [
            x >= 0,
            cp.sum(x, axis=0) <= 1,
            cp.sum(x, axis=1) <= 1,
        ]
        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve()
        assert cvxprob.status == "optimal"
        return x.value.clip(min=1e-5)

    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        return super().unflatten(self.get_allocation_flattened(throughputs),
                                 index)


class KSPolicyNormalized(KSPolicy):
    def get_allocation(self, unflattened_throughputs):
        throughputs, index = super().flatten(unflattened_throughputs)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        scale = 1.0 / throughputs.sum(axis=1)
        throughputs = throughputs * scale.reshape(m, 1)
        return super().unflatten(super().get_allocation_flattened(throughputs),
                                 index)

class FIFOPolicy:
    def __init__(self):
        self._allocation = {}
        self._queue = []

    def get_allocation(self, throughputs):
        # New Job ID; put on queue to schedule.
        job_id = None
        for job_id in throughputs:
            if job_id not in self._allocation and job_id not in self._queue:
                self._queue.append(job_id)

        # Old Job ID that has been removed; schedule job from queue.
        job_ids = list(self._allocation.keys())
        for job_id in job_ids:
            if job_id not in throughputs:
                worker_id = self._allocation[job_id]
                del self._allocation[job_id]
                if len(self._queue) > 0:
                    job_id_to_schedule = self._queue.pop(0)
                    self._allocation[job_id_to_schedule] = worker_id

        worker_ids_seen = set()
        for job_id in self._allocation:
            worker_id = self._allocation[job_id]
            worker_ids_seen.add(worker_id)

        job_ids = list(throughputs.keys())
        if len(job_ids) > 0:
            job_id = job_ids[0]
            for worker_id in throughputs[job_id]:
                if worker_id not in worker_ids_seen:
                    if len(self._queue) > 0:
                        job_id_to_schedule = self._queue.pop(0)
                        self._allocation[job_id_to_schedule] = worker_id

        allocation = {}
        for job_id in throughputs:
            allocation[job_id] = {}
            for worker_id in throughputs[job_id]:
                if job_id in self._allocation and self._allocation[job_id] == worker_id:
                    allocation[job_id][worker_id] = 1.0
                else:
                    allocation[job_id][worker_id] = 0.0
        return allocation


def get_num_steps_to_run(job_id, worker_type):
    return 1

def read_trace(trace_filename):
    timestamps_and_jobs = []
    # Trace file is expected to be in the following format:
    # <timestamp at which job is enqueued> <tab> <job_type> <tab> <command> <tab> <duration> <tab> <number of times to run command>.
    with open(trace_filename, 'r') as f:
        for line in f.read().strip().split('\n'):
            [timestamp, job_type, command, duration, num_steps] = line.split('\t')
            job_id = None
            job_type = job_type
            duration = float(duration)
            timestamp = int(timestamp)
            num_steps = int(num_steps)
            timestamps_and_jobs.append(
                (timestamp,
                 job.Job(job_id, job_type, command, num_steps, duration)))
    timestamps_and_jobs.sort(key=lambda x: x[0])
    return timestamps_and_jobs

def main(trace_filename, policy_name, worker_types, num_workers,
         normalizing_worker_type, sleep_seconds, emulate, throughputs_directory):
    prev_timestamp = None
    policy = None
    if policy_name == "isolated":
        policy = IsolatedPolicy()
    elif policy_name == "ks":
        policy = KSPolicy()
    elif policy_name == "ks_normalized":
        policy = KSPolicyNormalized()
    elif policy_name == "fifo":
        policy = FIFOPolicy()
    else:
        raise Exception("Unknown policy!")
    s = scheduler.Scheduler(policy, get_num_steps_to_run,
                            emulate=emulate,
                            normalizing_worker_type=normalizing_worker_type,
                            throughputs_directory=throughputs_directory)

    if emulate:
        for i in range(num_workers):
            worker_type = "dummy_worker"
            if worker_types is not None:
                worker_type = worker_types[i]
            s._register_worker_callback(
                worker_type=worker_type,
                ip_addr=None, port=None,
                devices=None)

    start = time.time()
    for (timestamp, job) in read_trace(trace_filename):
        if not emulate:
            if prev_timestamp is not None:
                time.sleep(timestamp - prev_timestamp)
            prev_timestamp = timestamp
            job_id = s.add_job(job)
        else:
            s.add_to_event_queue(s.add_job, [job], timestamp)

    if emulate:
        s.start_scheduling_thread()

    while not s.is_done():
        time.sleep(sleep_seconds)

    if not emulate:
        print("Total time taken: %.2f seconds" % (time.time() - start))
    s.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Execute a trace"
    )
    parser.add_argument('-t', "--trace_filename", type=str, required=True,
                        help="Trace filename")
    parser.add_argument("--policy_name", type=str, default="isolated",
                        help="Policy to use: fifo|isolated|ks|ks_normalized")
    parser.add_argument('-w', "--worker_types", type=str, nargs='+',
                        help="Worker types: [k80|p100|v100]+")
    parser.add_argument('-n', "--num_workers", type=int, default=None,
                        help="Number of workers to use for scheduling jobs (in emulation mode)")
    parser.add_argument('-s', "--sleep_seconds", type=float, default=0.1,
                        help="Number of seconds to sleep when waiting for all" \
                             "jobs to complete")
    parser.add_argument('--emulate', action='store_true',
                        help="Emulate execution of jobs")
    parser.add_argument("--throughputs_directory", type=str, default=None,
                        help="Directory with throughput measurements")
    args = parser.parse_args()

    if args.worker_types is not None:
        assert args.num_workers is None, \
            "num_workers shouldn't be specified when worker_types is specified"
        args.num_workers = len(args.worker_types)
        args.normalizing_worker_type = args.worker_types[0]
    main(args.trace_filename, args.policy_name, args.worker_types, args.num_workers,
         args.normalizing_worker_type, args.sleep_seconds, args.emulate,
         args.throughputs_directory)
