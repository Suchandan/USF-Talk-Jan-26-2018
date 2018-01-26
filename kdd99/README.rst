This code implements a simple two layer neural network with Tensorflow using the Inference-Loss-Training pattern.

## Getting Started

1. Split data into test/train/validation
2. Convert CSV data to TFRecord data
3. Train the model and print statistics.

or simply execute this code

```python
cd kdd99

python make_data.py
python make_tfrecords.py --directory ./data/split_data
python train_model.py
```