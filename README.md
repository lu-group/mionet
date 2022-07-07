# MIONet: Learning multiple-input operators via tensor product

The source code for the paper [Pengzhan Jin, Shuai Meng, and Lu Lu. "MIONet: Learning multiple-input operators via tensor product." arXiv preprint arXiv:2202.06137 (2022)](https://arxiv.org/abs/2202.06137).

## Code

- Data Generation
    - [ODE system](data/ODE_system.py)
    - [Diffusion-Reaction system](data/DR_system.py)
    - [Advection-Diffusion system](data/ADVD_system.py)
- Training
    - [MIONet training](training/MIONet_training.py)
    - [MIONet periodic training](training/mionet_periodic/main.py)
 
## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{jin2022mionet,
  title={MIONet: Learning multiple-input operators via tensor product},
  author={Jin, Pengzhan and Meng, Shuai and Lu, Lu},
  journal={arXiv preprint arXiv:2202.06137},
  year={2022}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
