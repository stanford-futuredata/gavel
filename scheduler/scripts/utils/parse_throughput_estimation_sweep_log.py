import argparse
import re

class Experiment:
    def __init__(self, experiment_id, params):
        self._experiment_id = experiment_id
        self._cluster_spec = params['cluster_spec']
        self._policy = params['policy']
        self._seed = int(params['seed'])
        self._lam = -1
        self._num_total_jobs = -1
        self._profiling_percentage = -1
        self._num_reference_models = -1
        self._average_jct = None
        self._utilization = None
        self._makespan = None
        self._total_cost = None
        self._num_SLO_violations = None

        if 'lam' in params:
            self._lam = float(params['lam'])
        if 'num_total_jobs' in params:
            self._num_total_jobs = int(params['num_total_jobs'])
        if 'profiling_percentage' in params:
            self._profiling_percentage = float(params['profiling_percentage'])
        if 'num_reference_models' in params:
            self._num_reference_models = int(params['num_reference_models'])

    def update_results(self, results):
        self._average_jct = float(results['average JCT'])
        self._utilization = float(results['utilization'])
        if 'makespan' in results:
            self._makespan = float(results['makespan'])
        if 'total_cost' in results:
            # Remove $
            self._total_cost = float(results['total_cost'][1:])
        if 'num_SLO_violations' in results:
            self._num_SLO_violations = int(results['num_SLO_violations'])

def print_experiments(experiments):
    print('Experiment ID,Cluster Spec,Policy,Seed,Lambda,Total Jobs,'
          'Profiling Percentage,Num Reference Models,Average JCT,'
          'Utilization,Makespan,Total Cost')

    all_experiment_ids = sorted(list(experiments.keys()))
    for experiment_id in all_experiment_ids:
        experiment = experiments[experiment_id]

        if experiment._average_jct is None:
            print('%d,%s,%s,%d,%f,'
                  '%d,%f,%d' % (experiment_id,
                                experiment._cluster_spec,
                                experiment._policy,
                                experiment._seed,
                                experiment._lam,
                                experiment._total_jobs,
                                experiment._profiling_percentage,
                                experiment._num_reference_models))
        else:
            print('%d,%s,%s,%d,%f,%d,%f,%d,'
                  '%f,%f,%f,%f,%d' % (experiment_id,
                                      experiment._cluster_spec,
                                      experiment._policy,
                                      experiment._seed,
                                      experiment._lam,
                                      experiment._num_total_jobs,
                                      experiment._profiling_percentage,
                                      experiment._num_reference_models,
                                      experiment._average_jct,
                                      experiment._utilization,
                                      experiment._makespan,
                                      experiment._total_cost,
                                      experiment._num_SLO_violations))

def main(args):
    experiments = {}
    with open(args.log_file, 'r') as f:
        for line in f:
            # Search for the experiment ID.
            m = re.search('\[Experiment ID: *(\d+)\]', line)
            if m:
                experiment_id = int(m.group(1))
            else:
                continue

            # Search for the configuration.
            m = re.search('Configuration: (.*)', line)
            if m:
                configuration = m.group(1).split(',')
                params = {}
                for param in configuration:
                    key, value = param.split('=')
                    params[key.strip()] = value.strip()
                experiments[experiment_id] = Experiment(experiment_id, params)
                continue

            # Search for the results.
            m = re.search('Results: (.*)', line)
            if m:
                results = m.group(1).split(',')
                params = {}
                for param in results:
                    key, value = param.split('=')
                    params[key.strip()] = value.strip()
                experiments[experiment_id].update_results(params)
    print_experiments(experiments)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Parse throughput estimation sweep log')
    parser.add_argument('-l', '--log_file', type=str, required=True,
                        help='Log file to parse')
    args = parser.parse_args()
    main(args)
