import os


def read_all_throughputs(directory_name, worker_types=["k80", "p100", "v100", "v100_1_22"]):
    all_throughputs = {}

    directory = os.fsencode(directory_name)
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        worker_type = file_name.rstrip(".log")
        if worker_type in worker_types:
            with open(os.path.join(directory_name, file_name), 'r') as f:
                job_types = None
                job_throughputs = None
                for line in f:
                    # NOTE: Some of this parsing code is incredibly ugly, but
                    # it gets the job done :(
                    line = line.strip()
                    if line != "":
                        if line.startswith("("):
                            job_types = line.lstrip("(").rstrip(")").split(", ")
                        elif line.startswith("["):
                            throughputs = [
                                float(x) for x in line.lstrip("[").rstrip("]").split(", ")]
                        else:
                            try:
                                throughputs = [float(line)]
                            except:
                                job_types = [line]
                    else:
                        job_types = tuple(job_types)
                        if job_types not in all_throughputs:
                            all_throughputs[job_types] = {}
                        all_throughputs[job_types][worker_type] = tuple(throughputs)
    return all_throughputs
