# Weight-Averaged Sharpness-Aware Minimization (WASAM)

![Alt Text](.github/wasam.gif)

A minimum working example for incorporating [WASAM](https://arxiv.org/pdf/2202.00661.pdf) in an image classification pipeline implemented in PyTorch.
# Quickstart

Install packages by 

```pip install -r requirements.txt```

Then, you can run

```
cd example
python main.py
```

## Citation

If you find this repository useful, please consider citing the paper. 

```
@misc{kaddour2022questions,
      title={Questions for Flat-Minima Optimization of Modern Neural Networks}, 
      author={Jean Kaddour and Linqing Liu and Ricardo Silva and Matt J. Kusner},
      year={2022},
      eprint={2202.00661},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements

This codebase builds on other repositories:

* [(Adaptive) SAM Optimizer (PyTorch)](https://github.com/davda54/sam). 
* [A PyTorch implementation for PyramidNets](https://github.com/dyhan0920/PyramidNet-PyTorch.git)
* [Label Smoothing in PyTorch](https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch)

Thanks a lot to the authors of these!
