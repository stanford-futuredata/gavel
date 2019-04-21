import argparse
import datetime
import numpy as np

NUM_SECONDS_PER_DAY = (24 * 60 * 60)

commands_to_job_types = {
    'cd /home/keshavsanthanam/gpusched/workloads/pytorch/image_classification/imagenet && python3 main.py -j 4 -a resnet50 -b 64 /home/deepakn94/imagenet/': 'ResNet-50',
    'cd /home/keshavsanthanam/gpusched/workloads/pytorch/image_classification/cifar10 && python3 main.py --data_dir=/home/keshavsanthanam/data/cifar10': 'ResNet-18',
    'cd /home/keshavsanthanam/gpusched/workloads/pytorch/translation/ && python3 train.py -data /home/keshavsanthanam/data/translation/multi30k.atok.low.pt -proj_share_weight': 'Transformer',
    'cd /home/keshavsanthanam/gpusched/workloads/pytorch/recommendation/scripts/ml-20m && python3 train.py': 'Recommendation',
    'cd /home/keshavsanthanam/gpusched/workloads/pytorch/rl && python3 main.py --env PongDeterministic-v4 --workers 4 --amsgrad True': 'A3C',
    'cd /home/keshavsanthanam/gpusched/workloads/pytorch/language_modeling && python main.py --cuda --data /home/keshavsanthanam/data/wikitext-2': 'LM',
    'cd /home/keshavsanthanam/gpusched/workloads/pytorch/cyclegan && python3 cyclegan.py --dataset_path /home/keshavsanthanam/data/monet2photo --decay_epoch 0': 'CycleGAN',
}

def get_seconds_from_timedelta(timedelta):
    return (timedelta.days * NUM_SECONDS_PER_DAY +
            timedelta.seconds +
            timedelta.microseconds / 1000.0)

def to_datetime(s):
    format_string = '%Y-%m-%d %H:%M:%S.%f'
    return datetime.datetime.strptime(s, format_string)

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
    timedelta = latest_end_time - earliest_dispatch_time
    return get_seconds_from_timedelta(timedelta)


class Job:
    def __init__(self, job_id, job_type, dispatch_time):
        self._job_id = job_id
        self._job_type = job_type
        self._dispatch_time = dispatch_time
        self._start_times = []
        self._end_times = []
        self._worker_types = []
        self._steps = {}
        self._throughputs = {}

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

    @property
    def throughputs(self):
        return self._throughputs

    def add_start_time(self, start_time):
        self._start_times.append(start_time)

    def add_end_time(self, end_time):
        self._end_times.append(end_time)
    
    def add_worker_type(self, worker_type):
        self._worker_types.append(worker_type)

    def add_steps(self, worker_type, steps):
        if worker_type not in self._steps:
            self._steps[worker_type] = []
        self._steps[worker_type].append(steps)

    def add_throughput(self, worker_type, throughput):
        if worker_type not in self._throughputs:
            self._throughputs[worker_type] = []
        self._throughputs[worker_type].append(throughput)

    def verify(self):
        check = True
        check = check and len(self._start_times) == len(self._end_times)
        check = check and len(self._end_times) == len(self._worker_types)
        for worker_type in self._worker_types:
            check = check and (len(self._throughputs[worker_type]) ==
                               len(self._steps[worker_type]))
        return check

    def queueing_delay(self):
        total_queueing_delay = 0.0
        queue_time = self._dispatch_time
        for i in range(len(self._start_times)):
            dequeue_time = self._start_times[i]
            queueing_delay = (dequeue_time - queue_time)
            total_queueing_delay += get_seconds_from_timedelta(queueing_delay)
            queue_time = self._end_times[i]
        return total_queueing_delay 
    
    def completion_time(self):
        return get_seconds_from_timedelta(self._end_times[-1] -
                                          self._dispatch_time)

    def total_computation_time(self):
        total_computation_time = 0.0
        for (start_time, end_time) in zip(self._start_times, self._end_times):
            computation_time = end_time - start_time
            total_computation_time += \
                    get_seconds_from_timedelta(computation_time)
        return total_computation_time

    def __str__(self):
        s = ''
        s += 'Job ID %d\n' % (self._job_id)
        s += 'Job type: %s\n' % (self._job_type)
        s += 'Dispatch time: %s\n' % (str(self._dispatch_time))
        s += 'Micro-tasks:\n'
        per_worker_idx = {}
        for worker_type in self._steps:
            per_worker_idx[worker_type] = 0
        for i, (worker_type, start_time, end_time) in enumerate(zip(self._worker_types, self._start_times, self._end_times)):
            s += '\t%2d) [%4s] [%4d steps] %s - %s (%.3f)\n' % (i, worker_type, self._steps[worker_type][per_worker_idx[worker_type]], str(start_time), str(end_time), get_seconds_from_timedelta(end_time - start_time))
            s += '\t\tThroughput on worker type %4s: %.3f -> %.4f\n' % (worker_type, self._throughputs[worker_type][per_worker_idx[worker_type]], self._throughputs[worker_type][per_worker_idx[worker_type]+1])
            per_worker_idx[worker_type] += 1
        s += 'Final throughputs:\n'
        worker_types = [worker_type for worker_type in self._throughputs]
        for worker_type in sorted(worker_types):
            s += '\t[%4s] %.3f\n' % (worker_type, self._throughputs[worker_type][-1]) 
        s += 'Total queuing delay: %.3f\n' % (self.queueing_delay())
        s += 'Total computation time: %.3f\n' % (self.total_computation_time())
        s += 'Job completion time: %.3f\n' % (self.completion_time())
        return s

def get_job_types(trace_file):
    job_types = []
    with open(trace_file, 'r') as f:
        for line in f:
            command  = line.split('\t')[0]
            if command not in commands_to_job_types:
                raise ValueError(command)
            job_types.append(commands_to_job_types[command])
    return job_types

def process_log(log_file, job_types):
    jobs = {}
    with open(log_file, 'r') as f:
        for line in f:
            if '[Job dispatched]' in line:
                dispatch_time = to_datetime(line.split(']')[0])
                job_id = int(line.strip().split('Job ID:')[-1])
                jobs[job_id] = Job(job_id, job_types[job_id], dispatch_time)
            elif '[Micro-task scheduled]' in line:
                start_time = to_datetime(line.split(']')[0])
                job_id = int(line.strip().split('Job ID:')[1].split(',')[0])
                jobs[job_id].add_start_time(start_time)
            elif '[Micro-task succeeded]' in line:
                end_time = to_datetime(line.split(']')[0])
                job_id = int(line.strip().split('Job ID:')[1].split(',')[0])
                jobs[job_id].add_end_time(end_time)
            elif '[DEBUG]' in line and ('throughput' in line or 'steps' in line):
                job_id = \
                        int(line.strip().split('Job')[-1].strip().split(' ')[0])
                worker_type = \
                        line.strip().split('worker type')[-1].split(':')[0].strip()
                if 'throughput' in line:
                    jobs[job_id].add_worker_type(worker_type)
                    throughputs = line.strip().split(':')[1].split('->')
                    if worker_type not in jobs[job_id].throughputs:
                        jobs[job_id].add_throughput(worker_type,
                                                    float(throughputs[0]))
                    jobs[job_id].add_throughput(worker_type,
                                                float(throughputs[-1]))
                elif 'steps' in line:
                    steps = line.strip().split(':')[1].split('->')
                    if worker_type not in jobs[job_id].steps:
                        jobs[job_id].add_steps(worker_type, float(steps[0]))
                    jobs[job_id].add_steps(worker_type, float(steps[-1]))

    for job_id in jobs:
        assert jobs[job_id].verify()

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

def main(args):
    job_types = get_job_types(args.trace_file)
    process_log(args.log_file, job_types)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_file', type=str, required=True,
                        help='Log file')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    args = parser.parse_args()
    main(args)
