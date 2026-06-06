# Powerful Switchback Experiments -- Or Not? (Simulation Code)

This directory contains the companion simulation code for the paper **"Powerful Switchback Experiments -- Or Not?"**. It provides a Jupyter notebook and utility scripts to validate the closed-form variance formula for the Average Treatment Effect (ATE) estimator in switchback experiments using simulated dummy data.

## Contents

* `switchback_power_paper_simulations.ipynb`: The main Jupyter notebook containing the One-Factor-At-a-Time (OFAT) simulation sweeps and visualizations.
* `sim_utils/sim_utils.py`: Python utility script containing the core Data Generating Process (DGP), variance component estimation (Method of Moments), and Monte Carlo simulation logic.

## Requirements

To run the notebook, you will need Python 3 and the following packages:
* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `seaborn`

You can install them via pip:
```bash
pip install -r requirements.txt
```

## Usage

1. Open and run `switchback_power_paper_simulations.ipynb` in Jupyter.
2. The notebook will execute the simulations and save the generated figures to a local `./output/` directory.