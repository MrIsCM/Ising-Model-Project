# Ising Model Simulation

This repository contains an implementation of the 2D Ising Model developed as part of a university project. It includes both CPU and GPU-accelerated versions of the simulation to study phase transitions in ferromagnetic systems. The code is optimized using **Numba** and **PyTorch (CUDA)** to enable efficient performance on larger lattices.

## Project Structure

- **`GPU_optimization.ipynb`**  
  This Jupyter notebook explores the GPU-accelerated implementation of the Ising Model using **PyTorch** with **CUDA** support. It is primarily intended for performance benchmarking and visualization of the model's behavior under different configurations.

- **`Ising_Model_Fast.py`**  
  This module contains all the core functions required for the Ising Model simulation. The functions are optimized using **Numba** to significantly speed up the computational performance on the CPU. This file serves as the core library of the project.

- **`main.py`**  
  This script executes the main logic of the simulation, leveraging the optimized functions from `Ising_Model_Fast.py`. It includes setup for the lattice, the Metropolis algorithm, and relevant data collection and visualization.

## Features

- Support for 2D lattices
- Metropolis-Hastings algorithm for spin updates
- Temperature sweep to observe phase transitions
- GPU acceleration using PyTorch (optional)
- CPU optimization using Numba
- Modular and reusable code structure

## Requirements

To run the code, make sure you have the following installed:

- Python 3.8+ (I )
- [NumPy](https://numpy.org/)
- [Numba](https://numba.pydata.org/)
- [Matplotlib](https://matplotlib.org/) (for optional plotting)
- [PyTorch](https://pytorch.org/) with CUDA (for GPU acceleration)
- [SciPy](https://scipy.org/)
- [Imageio](https://pypi.org/project/imageio/) (used to produce Gifs rapidely)

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

Make sure you have a compatible GPU and CUDA drivers installed if you want to run the GPU-accelerated version.