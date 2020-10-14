import os
import subprocess
import utils


def prune_logs(old_directory, new_directory):
    logfile_paths = utils.get_logfile_paths(old_directory)
    for logfile_path_and_metadata in logfile_paths:
        logfile_path = logfile_path_and_metadata[-1]
        new_logfile_path = logfile_path.replace(old_directory, new_directory)
        new_logfile_directory = os.path.dirname(new_logfile_path)
        os.makedirs(new_logfile_directory, exist_ok=True)
        print(logfile_path, new_logfile_path)
        subprocess.call("tail -n 10000 %s > %s" % (logfile_path, new_logfile_path),
                        shell=True)


if __name__ == '__main__':
    prune_logs(old_directory="/future/u/deepakn/gavel/logs/cluster_sweep_continuous_jobs_final",
               new_directory="/lfs/1/deepakn/gavel/scheduler/logs/cluster_sweep_continuous_jobs_final")
