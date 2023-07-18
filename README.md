## On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay

This repo contains the official PyTorch implementation of the NeurIPS'21 paper  

**On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay**  
[Ekaterina Lobacheva](https://tipt0p.github.io/)\*, 
[Maxim Kodryan](https://scholar.google.com/citations?user=BGVWciMAAAAJ&hl=en)\*, 
[Nadezhda Chirkova](https://nadiinchi.github.io/), 
[Andrey Malinin](https://scholar.google.com/citations?user=PnSFqO0AAAAJ&hl=en), 
[Dmitry Vetrov](https://scholar.google.com/citations?user=7HU0UoUAAAAJ&hl=en)

[arXiv](https://arxiv.org/abs/2106.15739) / 
[openreview](https://openreview.net/forum?id=B6uDDaDoW4a) /
[short poster video](https://nips.cc/virtual/2021/poster/26875) / 
[long talk](https://media.mis.mpg.de/mml/2021-12-02/) / 
[bibtex](https://tipt0p.github.io/papers/periodic_behavior_neurips21.txt)

## Abstract
<div align="justify">
<img align="right" width=35% src="https://tipt0p.github.io/images/periodic_behavior_neurips21.png" />
Training neural networks with batch normalization and weight decay has become a
common practice in recent years. In this work, we show that their combined use
may result in a surprising periodic behavior of optimization dynamics: the training
process regularly exhibits destabilizations that, however, do not lead to complete
divergence but cause a new period of training. We rigorously investigate the
mechanism underlying the discovered periodic behavior from both empirical and
theoretical points of view and analyze the conditions in which it occurs in practice.
We also demonstrate that periodic behavior can be regarded as a generalization
of two previously opposing perspectives on training with batch normalization and
weight decay, namely the equilibrium presumption and the instability presumption.
</div>


## Code

**Environment**
```(bash)
conda env create -f cycle_env.yml
```

**Example usage**  
To obtain one of the lines in Figure 2 in the paper:
1. Run script run_train_and_test.py to train and compute metrics (in the presented form, it trains a scale-invariant ConvNet on CIFAR-10 using SGD with learning rate 0.01 and weight decay 0.001 without data augmentation).
2. Use notebook Plots.ipynb to look at the results.

**Main parameters**  
To replicate other results from the paper, vary the parameters in run_train_and_test.py:
- dataset: CIFAR10 or CIFAR100
- to train fully scale-invariant networks use models ConvNetSI/ResNet18SI, fix_noninvlr = 0.0 (learning rate for not scale invariant parameters), and initscale = 10. (norm of the last layer weight matrix)
- to train all network parameters use models ConvNetSIAf/ResNet18SIAf, fix_noninvlr = -1 and initscale = -1
- to turn on the momentum use a non-zero value for it in params
- to turn on data augmentation delete noaug option from add_params
- to use full-batch GD add option fbgd to add_params
- to train a network with a fixed norm of scale-invariant parameters add option fix_si_pnorm to add_params and use a positive value for fix_si_pnorm_value in params
- you can change the learning rate, weight decay, and network width factor (num_channels in params)

## Attribution

Parts of this code are based on the following repositories:
- [Rethinking Parameter Counting: Effective Dimensionality Revisted](https://github.com/g-benton/hessian-eff-dim). Wesley Maddox, Gregory Benton, and Andrew Gordon Wilson.

## Citation

If you found this code useful, please cite our paper
```
@inproceedings{lobacheva2021periodic,
    title = {On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay},
    author = {Ekaterina Lobacheva and Maxim Kodryan and Nadezhda Chirkova and Andrey Malinin and Dmitry Vetrov},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2021}
    url = {https://openreview.net/forum?id=B6uDDaDoW4a},
}
```
