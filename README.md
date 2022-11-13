# MIONet: Learning multiple-input operators via tensor product

The source code for the paper [P. Jin, S. Meng, & L. Lu. MIONet: Learning multiple-input operators via tensor product. *SIAM Journal on Scientific Computing*, 44(6), A3490–A3514, 2022](https://doi.org/10.1137/22M1477751).

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
  title   = {MIONet: Learning multiple-input operators via tensor product},
  author  = {Jin, Pengzhan and Meng, Shuai and Lu, Lu},
  journal = {SIAM Journal on Scientific Computing},
  volume  = {44},
  number  = {6},
  pages   = {A3490-A3514},
  year    = {2022},
  doi     = {10.1137/22M1477751}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
