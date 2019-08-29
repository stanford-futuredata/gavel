import argparse
import re

class Experiment:
    def __init__(self, experiment_id, config):
        self._experiment_id = experiment_id
        self._policy = config['policy']
        self._seed = int(config['seed'])
        self._lam = float(config['lam'])
        self._completion_algo = config['completion_algo']
        if 'measurement_percentage' in config:
            self._measurement_percentage =\
                float(config['measurement_percentage'])
        elif 'drop_prob' in config:
            self._drop_prob = float(config['drop_prob'])
        else:
            raise ValueError('Could not find measurement_percentage or '
                             'drop_prob in config')
        self._average_jct = None
        self._utilization = None

    def store_results(self, results):
        self._average_jct = float(results['average JCT'])
        self._utilization = float(results['utilization'])

    def __str__(self):
        if hasattr(self, '_measurement_percentage'):
            x = self._measurement_percentage
        elif hasattr(self, '_drop_prob'):
            x = self._drop_prob
        else:
            raise ValueError('Could not find measurement_percentage or '
                             'drop_prob')

        if not self._average_jct and not self._utilization:
            return '%d,%s,%f,%s,%f,%d,' % (self._experiment_id,
                                           self._policy,
                                           self._lam,
                                           self._completion_algo,
                                           x,
                                           self._seed)
        else:
            return '%d,%s,%f,%s,%f,%d,%f,%f' % (self._experiment_id,
                                                self._policy,
                                                self._lam,
                                                self._completion_algo,
                                                x,
                                                self._seed,
                                                self._average_jct,
                                                self._utilization)

def group_experiments_by_completion_algos(experiments):
    exp_by_algo = {}
    for experiment in experiments:
        if experiment._completion_algo not in exp_by_algo:
            exp_by_algo[experiment._completion_algo] = []
        exp_by_algo[experiment._completion_algo].append(experiment)
    return exp_by_algo



def plot(exp_by_algo, measurement_percentage,
         completion_algos=['BMF', 'PMF', 'SVT']):
    for completion_algo in completion_algos:
        x = []
        y = []
        for experiment in exp_by_algo[completion_algo]:
            x.append(experiment._lam)
            y.append(experiment._average_jct)
        plt.plot(x, y, label=completion_algo)

def main(args):
    experiments = {}
    with open(args.log_file) as f:
        for line in f:
            result = re.search('\[Experiment ID: [0-9]*\]', line)
            if result is None:
                continue
            experiment_id = int(result.group()[:-1].split(':')[-1])
            try:
                idx = line.index('Configuration')
                line = line[idx:].strip()
                params = [param.split('=') for param in line.split(',')[1:]]
                config = {}
                for (key, value) in params:
                    config[key.strip()] = value.strip()
                experiments[experiment_id] = Experiment(experiment_id, config)
                continue
            except ValueError:
                pass
            try:
                idx = line.index('Results')
                line = line[idx:].strip().split(':')[1]
                params = [param.split('=') for param in line.split(',')]
                results = {}
                for (key, value) in params:
                    results[key.strip()] = value.strip()
                experiments[experiment_id].store_results(results)
            except ValueError as e:
                print(e)

    """
    algo_wins = {}
    all_lams = set()
    all_seeds = set()
    all_completion_algos = set()
    experiment_index = {}
    for experiment_id in experiments:
        if experiments[experiment_id]._measurement_percentage == 1.0:
            continue
        all_lams.add(experiments[experiment_id]._lam)
        all_seeds.add(experiments[experiment_id]._seed)
        all_completion_algos.add(experiments[experiment_id]._completion_algo)

    for lam in all_lams:
        experiment_index[lam] = {}
        for seed in all_seeds:
            experiment_index[lam][seed] = {}
            for completion_algo in all_completion_algos:
                experiment_index[lam][seed][completion_algo] = None
                for experiment_id in experiments:
                    experiment = experiments[experiment_id]
                    if (experiment._lam == lam and experiment._seed == seed and
                        experiment._completion_algo == completion_algo):
                        experiment_index[lam][seed][completion_algo] =\
                                experiment
    for lam in all_lams:
        for seed in all_seeds:
            results = []
            best_algo = None
            best_jct = float('inf')
            for completion_algo in all_completion_algos:
                experiment = experiment_index[lam][seed][completion_algo]
                if experiment is None or experiment._average_jct is None:
                    continue
                elif best_algo is None or experiment._average_jct < best_jct:
                    best_algo = completion_algo
                    best_jct = experiment._average_jct
            if best_algo is not None:
                if best_algo not in algo_wins:
                    algo_wins[best_algo] = 0
                algo_wins[best_algo] += 1
    import json
    print(json.dumps(algo_wins, indent=4))
    """
    if args.drop_prob:
        x = 'Drop Probability'
    else:
        x = 'Measurement Percentage'
    print('Experiment ID,Policy,Lambda,Completion Algorithm,'
          '%s,Seed,Average JCT,Utilization' % (x))
    for experiment_id in sorted(list(experiments.keys())):
        print(experiments[experiment_id])

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Parse throughput estimation sweep log')
    parser.add_argument('-l', '--log_file', type=str, required=True,
                        help='Log file to parse')
    parser.add_argument('-d', '--drop_prob', action='store_true',
                        default=False,
                        help='Parse drop probability')
    args = parser.parse_args()
    main(args)
