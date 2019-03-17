import argparse
import numpy
import random


models = [
    "vgg16",
    "nmt",
    "inception3",
    "vae",
    "resnet50",
    "lm_large",
]


def generate_trace(arrival_process, lam, trace_size):
    if arrival_process == "poisson":
        assert(lam is not None)
        inter_arrival_times = list(numpy.random.poisson(lam, trace_size-1))
        arrival_times = [0]
        for inter_arrival_time in inter_arrival_times:
            arrival_times.append(arrival_times[-1] + inter_arrival_time)
    elif arrival_process == "constant":
        arrival_times = [0] * trace_size
    else:
        raise Exception("Invalid arrival process")

    for arrival_time in arrival_times:
        random_idx = random.randint(0, len(models)-1)
        model_name = models[random_idx]
        duration = 10 ** random.uniform(0, 4)  # this is in minutes.
        duration *= 60
        # Split into 200 steps.
        duration /= 200.
        num_epochs = 200
        cmd = f"echo \"{model_name}\""
        print(f"{arrival_time}\t{model_name}\t{cmd}\t{duration}\t{num_epochs}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a trace"
    )
    parser.add_argument('-a', "--arrival_process", type=str, required=True,
                        help="Arrival process: poisson|constant")
    parser.add_argument('-l', "--lam", type=float, default=None,
                        help="Lambda parameter for Poisson arrival process")
    parser.add_argument('-s', "--trace_size", type=int, required=True,
                        help="Trace size")
    args = parser.parse_args()

    generate_trace(args.arrival_process, args.lam, args.trace_size)
