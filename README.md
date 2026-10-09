# NumPy MLP from scratch

A learning project implementing a multilayer perceptron in NumPy: manual backpropagation, ReLU, softmax cross-entropy, mini-batch Adam and early stopping. Experiments use synthetic data, MNIST-1D and MNIST handwritten digits.

## Development

I develop this project in my spare time to understand neural networks by implementing them myself. I plan to add a CNN and compare my implementations with equivalent models in established libraries, starting with PyTorch. These additions are planned work.

## Run

Use Python 3.10 or later. Place `nn_from_scratch_clean.ipynb` and `requirements_nn.txt` in the same directory, then install the listed dependencies in your Python environment:

```text
python -m pip install -r requirements_nn.txt
```

Open the notebook in Jupyter or VS Code, select that environment as the kernel, and run the cells in order. The dependency file includes the kernel; use your existing notebook interface. Data downloads automatically on the first run and is cached in `data/` relative to the notebook's working directory. If downloads are unavailable, put the four MNIST `.gz` files and `mnist1d_data.pkl` in that directory.

The MNIST-1D grid search trains 30 models. The MNIST experiment trains on the full training dataset with a validation split and early stopping. Runtime depends on the computer and NumPy's linear algebra library.

## Inspect predictions

The last two cells sample nine test digits with the true and predicted class, then follow one image through the layer dimensions and plot the softmax output for all ten classes.

Change `seed` in `show_random_predictions(...)` to sample different images. Change `index` in `show_prediction_steps(...)` to inspect another test image. Green/red titles indicate correct/incorrect predictions. Softmax outputs have not been calibrated.

## Scope and sources

The original MLP structure and training algorithm are retained. Small corrections add a stable log-softmax loss, preserve the actual best validation checkpoint, keep all output classes in the confusion matrix, and use the fixed 255 pixel scale for MNIST. An unused squared-error helper was removed; this implementation supports softmax cross-entropy classification. The incomplete PyTorch sketch, diagnostic one-off cells and the extra MNIST-1D refit were omitted from this copy.

Training curves include every epoch, whereas final evaluation uses the restored best validation checkpoint. The original MNIST learning rate is retained and can produce unstable late-training curves; learning-rate comparisons are a useful next experiment. The MNIST-1D search is a single-run example, not a statistically robust ranking of hyperparameters.

The MNIST download and decoding code follows the [NumPy MNIST tutorial](https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/). MNIST-1D is loaded through the `mnist1d` package. The method named `Xavier_Glorot_init` retains the original custom scale `sqrt(4 / (fan_in + fan_out))`; it is not standard normal Glorot initialization.

The notebook includes executed outputs. Exact training results can differ between environments. The experiments use a fixed initial seed, validation-based model selection and held-out test evaluation.
