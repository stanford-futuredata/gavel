import os
import signal
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import utils

gpu_processes = utils.get_gpu_processes()
for gpu_id in gpu_processes:
    for pid in gpu_processes[gpu_id]:
        print('Killing process %d on GPU %d' % (pid, gpu_id))
        os.kill(pid, signal.SIGKILL)
