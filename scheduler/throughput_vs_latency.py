import argparse

import job
import policies
import scheduler

def get_policy(policy_name):
    if policy_name == "isolated":
        policy = policies.IsolatedPolicy()
    elif policy_name == "max_min_fairness":
        policy = policies.MaxMinFairnessPolicy()
    elif policy_name == "max_min_fairness_packed":
        policy = policies.MaxMinFairnessPolicyWithPacking()
    elif policy_name == "min_total_duration":
        policy = policies.MinTotalDurationPolicy()
    elif policy_name == "min_total_duration_packed":
        policy = policies.MinTotalDurationPolicyWithPacking()
    elif policy_name == "fifo":
        policy = policies.FIFOPolicy()
    else:
        raise Exception("Unknown policy!")
    return policy

def parse_trace(trace_file):
    jobs = []
    arrival_times = []
    with open(trace_file, 'r') as f:
        for line in f:
            job_type, command, num_steps_arg, total_steps, arrival_time, scale_factor = \
                    line.split('\t')
            jobs.append(job.Job(job_id=None,
                                job_type=job_type,
                                command=command,
                                num_steps_arg=num_steps_arg,
                                total_steps=int(total_steps),
                                duration=None,
                                scale_factor=int(scale_factor)))
            arrival_times.append(int(arrival_time))
    return jobs, arrival_times


def main():
    # TODO: convert to command line arguments
    schedule_in_rounds = False
    throughputs_file = 'combined_throughputs.json'
    num_v100s = 28
    policy_names = ['max_min_fairness', 'max_min_fairness_packed'] 
    ratios = [
            {'v100': 1, 'p100': 0, 'k80': 0},
            #{'v100': 1, 'p100': 1, 'k80': 0},
            #{'v100': 1, 'p100': 1, 'k80': 1},
            #{'v100': 2, 'p100': 1, 'k80': 0},
        ]
    lams = [256]
    for ratio in ratios:
        cluster_spec = {}
        total_gpu_fraction = sum([ratio[gpu_type] for gpu_type in ratio])
        for gpu_type in ratio:
            cluster_spec[gpu_type] = int(ratio[gpu_type] / total_gpu_fraction * num_v100s)
        output_file = ('latency_vs_throughput_cluster_ratio_'
                       '%d_%d_%d.csv') % (ratio['v100'], ratio['p100'],
                                          ratio['k80'])
        with open(output_file, 'w') as f:
            f.write('# v100,# p100,# k80,Policy,Lambda,Utilization,Average JCT\n')
            for policy_name in policy_names:
                for lam in lams:
                    # TODO: can this be moved to previous for loop?
                    policy = get_policy(policy_name)
                    trace = 'traces/generated/microbenchmark/arrival_rate_%d.trace' % (lam)
                    jobs, arrival_times = parse_trace(trace)
                    sched = \
                        scheduler.Scheduler(policy,
                                            schedule_in_rounds=schedule_in_rounds,
                                            throughputs_file=throughputs_file,
                                            emulate=True)
                    sched.emulate(cluster_spec, arrival_times, jobs,
                                  ideal=False)
                    utilization = sched.get_cluster_utilization()
                    average_jct = sched.get_average_jct()
                    f.write('%d,%d,%d,%s,%d,%.3f,%.3f\n' % (cluster_spec['v100'],
                                                            cluster_spec['p100'],
                                                            cluster_spec['k80'],
                                                            policy.name,
                                                            lam,
                                                            utilization,
                                                            average_jct))
                    f.flush()


if __name__=='__main__':
    main()
