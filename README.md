# Metaparameter optimization in knowledge distillation

**Authors**: M. Gorpinich, O. Bakhteev, V. Strijov

This paper investigates the deep learning knowledge distillation problem. Knowledge distillation is a model parameter optimization problem that allows transferring information contained in the model with high complexity, called teacher, to the simpler one, called student. The paper proposes a generalization of the knowledge distillation optimization problem to optimize metaparameters by gradient descent. Metaparameters are the parameters of knowledge distillation optimization problem. The loss function is a sum of the classification term and cross-entropy between answers of the student model and teacher model. An assignment of optimal metaparameters for the distillation loss function is a computationally expensive task. We investigate the properties of optimization problem and methods to optimize and predict the regularization path of metaparameters. We analyze the trajectory of the metaparamets gradient-based optimization and approximate them using linear functions.  We evaluate the proposed method on the CIFAR-10 and Fashion-MNIST datasets and synthetic data.

## Requirements

```
Python >= 3.5.5
torch == 1.7.1
numpy == 1.18.5
tqdm == 4.59.0
matplotlib == 3.3.2
hyperopt == 0.2.5
scipy == 1.5.2
Pillow == 7.2.0
```

[requirements.txt](https://github.com/Intelligent-Systems-Phystech/Gorpinich-BS-Thesis/blob/master/requirements.txt)

## Main experiments:

[Synthetic data](https://github.com/Intelligent-Systems-Phystech/Gorpinich-BS-Thesis/blob/master/notebooks/synthetic_data_experiment.ipynb)

[CIFAR-10](https://github.com/Intelligent-Systems-Phystech/Gorpinich-BS-Thesis/blob/master/notebooks/cifar_data_experiment.ipynb)

[Fashion-MNIST](https://github.com/Intelligent-Systems-Phystech/Gorpinich-BS-Thesis/blob/master/notebooks/full_fmnist_data_experiment.ipynb)

