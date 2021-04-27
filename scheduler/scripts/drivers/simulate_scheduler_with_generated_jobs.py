import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import contextlib
import sys

import job
from job_id_pair import JobIdPair
from job_table import JobTable
import policies
import scheduler
import utils

def simulate(policy_name, throughputs_file, cluster_spec,
             lam, seed, interval, jobs_to_complete,
             fixed_job_duration, generate_multi_gpu_jobs,
             generate_multi_priority_jobs,
             simulate_steady_state, solver, debug,
             checkpoint_threshold, checkpoint_file,
             profiling_percentage, per_instance_type_prices_dir,
             available_clouds, assign_SLOs, enable_global_queue,
             num_gpus_per_server, output_trace_file_name,
             num_sub_problems):
    policy = utils.get_policy(policy_name, solver=solver, seed=seed)
    sched = scheduler.Scheduler(
                    policy,
                    throughputs_file=throughputs_file,
                    seed=seed,
                    time_per_iteration=interval,
                    simulate=True,
                    profiling_percentage=profiling_percentage,
                    per_instance_type_prices_dir=per_instance_type_prices_dir,
                    available_clouds=available_clouds,
                    assign_SLOs=assign_SLOs,
                    enable_global_queue=enable_global_queue,
                    num_sub_problems=num_sub_problems)

    cluster_spec_str = 'v100:%d|p100:%d|k80:%d' % (cluster_spec['v100'],
                                                   cluster_spec['p100'],
                                                   cluster_spec['k80'])
    current_time = datetime.datetime.now()
    print('[%s] Configuration: cluster_spec=%s, policy=%s, '
           'seed=%d, lam=%f, num_sub_problems=%d' % (
               current_time, cluster_spec_str, policy.name,
               seed, lam, num_sub_problems),
          file=sys.stderr)

    if lam == 0:
        num_total_jobs = len(jobs_to_complete)
    else:
        num_total_jobs = None

    sched.simulate(cluster_spec, lam=lam,
                   jobs_to_complete=jobs_to_complete,
                   fixed_job_duration=fixed_job_duration,
                   generate_multi_gpu_jobs=generate_multi_gpu_jobs,
                   generate_multi_priority_jobs=generate_multi_priority_jobs,
                   simulate_steady_state=simulate_steady_state,
                   num_total_jobs=num_total_jobs,
                   debug=debug,
                   checkpoint_threshold=checkpoint_threshold,
                   checkpoint_file=checkpoint_file,
                   num_gpus_per_server=num_gpus_per_server,
                   output_trace_file_name=output_trace_file_name)
    average_jct = sched.get_average_jct(jobs_to_complete)
    utilization = sched.get_cluster_utilization()
    total_cost = sched.get_total_cost()
    num_SLO_violations = sched.get_num_SLO_violations()
    lease_extension_freq = sched.get_num_lease_extensions()

    current_time = datetime.datetime.now()
    print('[%s] Results: average JCT=%f, utilization=%f, '
          'total_cost=$%.2f, num_SLO_violations=%d, '
          'lease_extension_frequency=%.2f%%' % (current_time,
                                                average_jct,
                                                utilization,
                                                total_cost,
                                                num_SLO_violations,
                                                lease_extension_freq),
          file=sys.stderr)

    sched.shutdown()

def main(args):
    throughputs_file = args.throughputs_file
    num_gpus = args.cluster_spec.split(':')
    cluster_spec = {
        'v100': int(num_gpus[0]),
        'p100': int(num_gpus[1]),
        'k80': int(num_gpus[2]),
    }
    num_gpus_per_server_split = args.num_gpus_per_server.split(':')
    num_gpus_per_server = {
        'v100': int(num_gpus_per_server_split[0]),
        'p100': int(num_gpus_per_server_split[1]),
        'k80': int(num_gpus_per_server_split[2]),
    }

    jobs_to_complete = set()
    for i in range(args.window_start, args.window_end):
        jobs_to_complete.add(JobIdPair(i, None))

    if args.verbose:
        simulate(args.policy, throughputs_file,
                 cluster_spec, args.lam, args.seed,
                 args.interval, jobs_to_complete,
                 args.fixed_job_duration,
                 args.generate_multi_gpu_jobs,
                 args.generate_multi_priority_jobs,
                 args.simulate_steady_state,
                 args.solver,
                 args.debug, args.checkpoint_threshold,
                 args.checkpoint_file,
                 args.profiling_percentage,
                 args.per_instance_type_prices_dir,
                 args.available_clouds,
                 args.assign_SLOs,
                 args.enable_global_queue,
                 num_gpus_per_server,
                 args.output_trace_file_name,
                 args.num_sub_problems)

    else:
        with open('/dev/null', 'w') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                simulate(args.policy, throughputs_file,
                         cluster_spec, args.lam, args.seed,
                         args.interval, jobs_to_complete,
                         args.fixed_job_duration,
                         args.generate_multi_gpu_jobs,
                         args.generate_multi_priority_jobs,
                         args.simulate_steady_state,
                         args.solver, args.debug,
                         args.checkpoint_threshold,
                         args.checkpoint_file,
                         args.profiling_percentage,
                         args.per_instance_type_prices_dir,
                         args.available_clouds,
                         args.assign_SLOs,
                         args.enable_global_queue,
                         num_gpus_per_server,
                         args.output_trace_file_name,
                         args.num_sub_problems)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Emulate scheduler with generated jobs')
    parser.add_argument('-c', '--cluster_spec', type=str, default='25:0:0',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--num_gpus_per_server', type=str, default='1:1:1',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    parser.add_argument('--num_sub_problems', type=int, nargs='+',
                        default=[1], help='Number of sub-problems')
    parser.add_argument('-s', '--window-start', type=int, default=0,
                        help='Measurement window start (job ID)')
    parser.add_argument('-e', '--window-end', type=int, default=5000,
                        help='Measurement window end (job ID)')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=utils.get_available_policies(),
                        help='Scheduler policy')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-i', '--interval', type=int, default=360,
                        help='Interval length (in seconds)')
    parser.add_argument('-l', '--lam', type=float, required=True,
                        help='Lambda for Poisson arrival rate')
    parser.add_argument('-f', '--fixed-job-duration', type=int, default=None,
                        help=('If set, fixes the duration of all jobs to the '
                              'specified value (in seconds)'))
    parser.add_argument('--throughputs_file', type=str,
                        default=('simulation_throughputs.json'),
                        help='Oracle throughputs file')
    parser.add_argument('-m', '--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('--simulate-steady-state', action='store_true',
                        default=False,
                        help=('If set, adds as many jobs as there are workers '
                              'before beginning the simulation.'))
    parser.add_argument('--solver', type=str, choices=['ECOS', 'GUROBI', 'SCS'],
                        default='ECOS', help='CVXPY solver')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Debug')
    parser.add_argument('--checkpoint_threshold', type=int, default=None,
                        help='Create checkpoint when this job ID comes in')
    parser.add_argument('--checkpoint_file', default=None,
                        help='Load checkpoint located at passed in checkpoint_file')
    parser.add_argument('--profiling_percentage', type=float, default=1.0,
                        help=('Percentage of co-located throughput values '
                              'measured for each job'))
    parser.add_argument('--num_reference_models', type=int,
                        default=len(JobTable),
                        help=('Number of reference models to use when '
                              'estimating throughputs'))
    parser.add_argument('--per_instance_type_prices_dir', type=str,
                        default=None,
                        help='Per-instance-type prices directory')
    parser.add_argument('--available_clouds', type=str, nargs='+',
                        choices=['aws', 'gcp', 'azure'],
                        default=['aws', 'gcp', 'azure'],
                        help='Clouds available to rent machines from')
    parser.add_argument('--assign_SLOs', action='store_true', default=False,
                        help='If set, assigns SLOs to each job')
    parser.add_argument('--enable_global_queue', action='store_true',
                        default=False,
                        help=('If set, schedules jobs regardless of '
                              'worker type'))
    parser.add_argument('--output_trace_file_name', type=str, default=None,
                        help='File to output generated trace to')

    args = parser.parse_args()
    main(args)
