# Introduction to Supervised Learning

This repository contains slides and an interactive notebook on supervised learning. In the slides the reader learns about what neural networks are, and how they work. In the notebook is a Python tutorial where the reader builds a supervised learning network to classify stars, galaxies and quasars from SDSS spectra.

## Installation & Setup

The notebooks rely only on `numpy`, `matplotlib`, and Jupyter. Any recent Python 3.9+ interpreter works, but the simplest way to keep dependencies isolated is to use a virtual environment:

```bash
cd "Machine Learning Tutorial"
python3 -m venv .venv
source .venv/bin/activate          # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy matplotlib jupyter
```

Once the dependencies install, launch Jupyter and open the tutorial:

```bash
jupyter lab notebooks/sdss_tutorial.ipynb
# or
jupyter notebook notebooks/sdss_tutorial.ipynb
```

All required SDSS sample data already lives under the `data/` folder, so you can run the notebook cells as-is. If you want to reset the environment later, simply remove the `.venv` directory and repeat the steps above.
