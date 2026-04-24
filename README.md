# Self-Pruning Neural Network

This repository contains a PyTorch implementation of a **Self-Pruning Neural Network**. The model is designed to dynamically identify and drop unnecessary connections during the training process using an L1 sparsity penalty on learnable gate scores.

By pruning redundant weights, the network can potentially reduce its memory footprint and improve inference speed while maintaining high predictive accuracy.

## Repository Structure

*   `self_pruning_nn.ipynb`: A Jupyter Notebook providing a step-by-step walkthrough of the code, including the custom `PrunableLinear` layer, data loading, training, and a visualization of the final gate value distribution.
*   `self_pruning_report.md`: A detailed technical report analyzing the model architecture, the mathematical mechanism behind the self-pruning gates, and the expected sparsity-vs-accuracy trade-offs.
*   `data/`: Directory where the CIFAR-10 dataset is automatically downloaded and stored.

## Requirements

To run the code in this repository, you need the following dependencies installed:

*   Python 3.8+
*   PyTorch
*   Torchvision
*   Pandas
*   NumPy
*   Matplotlib (for the Jupyter Notebook visualizations)

You can install the required packages using pip:
```bash
pip install torch torchvision pandas numpy matplotlib
```

## How to Run

To run the experiment and see the visualizations of the network's gate distributions, open the provided Jupyter Notebook:

```bash
jupyter notebook self_pruning_nn.ipynb
```

## How it Works

1.  **PrunableLinear Layer**: Instead of standard linear layers, the network's dense head uses custom layers equipped with `gate_scores`. 
2.  **Sigmoid Gates**: During the forward pass, a sigmoid function is applied to these gate scores to bind them between 0 and 1. The original weights are multiplied by these gate values.
3.  **L1 Sparsity Loss**: An L1 regularization penalty is applied to the gate values, controlled by a hyperparameter $\lambda$. As the model trains, it learns to push the gate scores of unimportant weights towards 0, effectively "pruning" them from the network.

## Documentation

For an in-depth analysis of the pruning mechanism and expected results, please refer to the [Technical Report](self_pruning_report.md) included in this repository.
