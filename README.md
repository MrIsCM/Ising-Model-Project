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


## Media

<div align="center">
  <img src="Simulations\Simulation_N100_T0.1\figures\demo.gif" alt="Ising Model Simulation Demo" width="400" loop="infinite"/>
  <br/>
  <b>Ising Model simulation (N=100, T=0.1)</b>
</div>

<details>
<summary>Note</summary>
Most browsers automatically loop GIFs. If your GIF does not loop, ensure it was exported with looping enabled. The <code>loop="infinite"</code> attribute is not standard HTML, but some Markdown renderers may support it.
</details>

Some graphs produced with the simulation data:

<div align="center">

<table>
    <tr>
        <td><img src="Simulations\Simulation_N2_T2\figures\energy_vs_T_N512.png" alt="Energy vs Temperature (N=512)" width="350"/></td>
        <td><img src="Simulations\Simulation_N2_T2\figures\magnetization_vs_T_N512.png" alt="Magnetization vs T (N=512)" width="350"/></td>
    </tr>
    <tr>
        <td align="center"><b>Energy vs Temperature (N=512)</b></td>
        <td align="center"><b>Magnetization vs T (N=512)</b></td>
    </tr>
</table>

</div>

<div align="center">

<table>
    <tr>
        <td><img src="Simulations\Simulation_N2_T2\figures\Cv_vs_T_N512.png" alt="Cv vs Temperature (N=512)" width="350"/></td>
        <td><img src="Simulations\Simulation_N2_T2\figures\loglog_alpha_N512.png" alt="Alpha parameter" width="350"/></td>
    </tr>
    <tr>
        <td align="center"><b>Heat capacity vs Temperature (N=512)</b></td>
        <td align="center"><b>&#945; parameter linear regression</b></td>
    </tr>
</table>

</div>


## Requirements

To run the code, make sure you have the following installed:

- Python 3.9+
- [NumPy](https://numpy.org/)
- [Numba](https://numba.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch](https://pytorch.org/) with CUDA (for GPU acceleration)
- [SciPy](https://scipy.org/)
- [Imageio](https://pypi.org/project/imageio/) (used to produce Gifs rapidely)

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

This command does not install PyTorch nor CUDA.