import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

NUM_SECONDS_PER_DAY = (24 * 60 * 60)

def overall_execution_time(jobs):
    earliest_dispatch_time = None
    latest_end_time = None
    for job_id in jobs:
        if (earliest_dispatch_time is None or
            jobs[job_id].dispatch_time < earliest_dispatch_time):
            earliest_dispatch_time = jobs[job_id].dispatch_time
        if (latest_end_time is None or
            jobs[job_id].end_times[-1] > latest_end_time):
            latest_end_time = jobs[job_id].end_times[-1]
    return latest_end_time - earliest_dispatch_time


class Job:
    def __init__(self, job_id, job_type, dispatch_time):
        self._job_id = job_id
        self._job_type = job_type
        self._dispatch_time = dispatch_time
        self._start_times = []
        self._end_times = []
        self._worker_types = []
        self._steps = {}

    @property
    def job_id(self):
        return self._job_id

    @property
    def dispatch_time(self):
        return self._dispatch_time

    @property
    def start_times(self):
        return self._start_times

    @property
    def end_times(self):
        return self._end_times

    @property
    def worker_types(self):
        return self._worker_types

    @property
    def steps(self):
        return self._steps

    def add_start_time(self, start_time):
        self._start_times.append(start_time)

    def add_end_time(self, end_time):
        self._end_times.append(end_time)

    def add_worker_type(self, worker_type):
        self._worker_types.append(worker_type)

    def verify(self):
        check = True
        check = check and len(self._start_times) == len(self._end_times)
        check = check and len(self._end_times) == len(self._worker_types)
        return check

    def queueing_delay(self):
        total_queueing_delay = 0.0
        queue_time = self._dispatch_time
        for i in range(len(self._start_times)):
            dequeue_time = self._start_times[i]
            queueing_delay = (dequeue_time - queue_time)
            total_queueing_delay += queueing_delay
            queue_time = self._end_times[i]
        return total_queueing_delay

    def completion_time(self):
        return self._end_times[-1] - self._dispatch_time

    def total_computation_time(self):
        total_computation_time = 0.0
        for (start_time, end_time) in zip(self._start_times, self._end_times):
            computation_time = end_time - start_time
            total_computation_time += computation_time
        return total_computation_time

    def __str__(self):
        s = ''
        s += 'Job ID %d\n' % (self._job_id)
        s += 'Job type: %s\n' % (self._job_type)
        s += 'Dispatch time: %s\n' % (str(self._dispatch_time))
        s += 'Micro-tasks:\n'
        job_metadata = zip(self._worker_types, self._start_times,
                           self._end_times)
        for i, (worker_type, start_time, end_time) in enumerate(job_metadata):
            s += '\t%2d) [%4s] %.3f - %.3f (%.3f)\n' % (i, worker_type,
                                                        start_time, end_time,
                                                        end_time - start_time)
        s += 'Total queuing delay: %.3f\n' % (self.queueing_delay())
        s += 'Total computation time: %.3f\n' % (self.total_computation_time())
        s += 'Job completion time: %.3f\n' % (self.completion_time())
        return s

def get_job_types(trace_file):
    job_types = []
    with open(trace_file, 'r') as f:
        for line in f:
            job_type = line.split('\t')[0]
            job_types.append(job_type)
    return job_types

def process_log(log_file, job_types):
    jobs = {}
    with open(log_file, 'r') as f:
        for line in f:
            if '[Job dispatched]' in line:
                dispatch_time = float(line.split(']')[0])
                job_id = int(line.strip().split('Job ID:')[-1])
                jobs[job_id] = Job(job_id, job_types[job_id], dispatch_time)
            elif '[Micro-task scheduled]' in line:
                start_time = float(line.split(']')[0])
                job_id = int(line.strip().split('Job ID:')[1].split(',')[0])
                worker_type = line.strip().split('Worker type: ')[-1]
                jobs[job_id].add_start_time(start_time)
                jobs[job_id].add_worker_type(worker_type)
            elif '[Micro-task succeeded]' in line:
                end_time = float(line.split(']')[0])
                job_id = int(line.strip().split('Job ID:')[1].split(',')[0])
                jobs[job_id].add_end_time(end_time)

    for job_id in jobs:
        assert jobs[job_id].verify()

    return jobs


def print_job_summary(jobs):
    for job_id in jobs:
        print(jobs[job_id])
        print('')
    job_completion_times = [jobs[job_id].completion_time() for job_id in jobs]
    queueing_delays = [jobs[job_id].queueing_delay() for job_id in jobs]
    computation_times = [jobs[job_id].total_computation_time() for job_id in jobs]
    print('Average job completion time: %.3f' % (np.mean(job_completion_times)))
    print('Average queuing delay: %.3f' % (np.mean(queueing_delays)))
    print(('Queueing delay / (Queueing delay + Computation time): '
          '%.3f') % (sum(queueing_delays) / (sum(queueing_delays) + sum(computation_times))))
    print('Overall execution time: %.3f' % (overall_execution_time(jobs)))


def plot_jct_cdf(jobs):
    min_x = 1e9
    max_x = 0
    for logfile in jobs:
        job_completion_times = [jobs[logfile][job_id].completion_time() for job_id in jobs[logfile]]
        num_bins = 20
        counts, bin_edges = np.histogram(job_completion_times, bins=num_bins,
                                         normed=True)
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[1:], cdf/cdf[-1], label=logfile)
        if bin_edges[1] < min_x:
            min_x = bin_edges[1]
        if bin_edges[-1] > max_x:
            max_x = bin_edges[-1]
    plt.xlabel('JCT (seconds)')
    #plt.axhline(y=.8, color='black', linestyle='--')
    #plt.axhline(y=.9, color='black', linestyle='--')
    plt.xlim((min_x, max_x))
    plt.legend(loc=0)
    plt.show()

def main(args):
    job_types = get_job_types(args.trace_file)
    jobs = {}
    for log_file in args.log_files:
        jobs[log_file] = process_log(log_file, job_types)
    plot_jct_cdf(jobs)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_files', type=str, nargs='+', required=True,
                        help='Log file')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    args = parser.parse_args()
    main(args)
