The code for the paper 
[**On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay**](https://arxiv.org/abs/2106.15739), NeurIPS'21

Create env:
```(bash)
conda env create -f cycle_env.yml
```

Example usage - how to obtain one of the lines at figure 2 in the paper:
1. Run script run_train_and_test.py to train and compute metrics (in the presented form it traines a scale-invariant ConvNet on CIFAR-10 dataset trained using standard SGD with learning rate 0.01 and weight decay 0.001 without data augmentation).
2. Use notebook Plots.ipynb to look at the results.

Main parameters to change in run_train_and_test.py to replicate other results:
- dataset: CIFAR10 or CIFAR100
- to train fully scale-invariant networks use models ConvNetSI/ResNet18SI, fix_noninvlr = 0.0 (learning rate for not scale invariant parameters) and initscale = 10. (norm of the last layer weight matrix)
- to train all network parameters use models ConvNetSIAf/ResNet18SIAf, fix_noninvlr = -1 and initscale = -1
- to turn on the momentum use a non-zero value for it in params
- to turn on data augmentation delete noaug option from add_params
- to use full-batch GD add option fbgd to add_params
- to train network with fixed norm of scale-invariant parameters add option fix_si_pnorm to add_params and use a positive value for fix_si_pnorm_value in params
- you can change learning rate, weight decay and network width factor (num_channels in params)
