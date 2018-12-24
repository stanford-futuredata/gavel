## Models

| Model                   | Throughput metric                                                              |
| ----------------------- | ------------------------------------------------------------------------------ |
| [ResNet-50](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) | Images / second             |
| [VGG16](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) | Images / second                 |
| [Inception v3](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) | Images / second          |
| [AlexNet](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) | Images / second               |
| [NMT](https://github.com/tensorflow/nmt)                                              | Words per second (in thousands) |
| [Language Model (small)](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)             | Words per second (in thousands) |
| [Language Model (medium)](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)             | Words per second (in thousands) |
| [Language Model (large)](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)             | Words per second (in thousands) |
| [VAE](https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW)                          | Images / second                 |

## measure\_throughput.py
```
usage: measure_throughput.py [-h] -o OUTPUT_FILE [-c DATA_DIR_CNN]
                             [-n DATA_DIR_NMT] [-r DATA_DIR_LM] [-m MINUTES]
                             [-d]

Measure throughput of co-located pairs of models

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        File to write results to
  -c DATA_DIR_CNN, --data_dir_cnn DATA_DIR_CNN
                        Data directory for CNNs
  -n DATA_DIR_NMT, --data_dir_nmt DATA_DIR_NMT
                        Data directory for NMT
  -r DATA_DIR_LM, --data_dir_lm DATA_DIR_LM
                        Data directory for language modeling RNN
  -m MINUTES, --minutes MINUTES
                        Number of minutes to run each model for
  -d, --debug           Run in debug mode
  ```

## measure\_nvprof\_metrics.py
```
usage: measure_nvprof_metrics.py [-h] -l LOG_DIR [-c DATA_DIR_CNN]
                                 [-n DATA_DIR_NMT] [-r DATA_DIR_LM]
                                 [-m NUM_STEPS] [-d]

Measure nvprof metrics for different models

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_DIR, --log_dir LOG_DIR
                        Log directory
  -c DATA_DIR_CNN, --data_dir_cnn DATA_DIR_CNN
                        Data directory for CNNs
  -n DATA_DIR_NMT, --data_dir_nmt DATA_DIR_NMT
                        Data directory for NMT
  -r DATA_DIR_LM, --data_dir_lm DATA_DIR_LM
                        Data directory for language modeling RNN
  -m NUM_STEPS, --num_steps NUM_STEPS
                        Number of steps to run each model for
  -d, --debug           Run in debug mode
  ```
