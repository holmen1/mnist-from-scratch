# MNIST from Scratch

A simple implementation of an MNIST digit classifier using NumPy

## Project Overview

This project demonstrates building a neural network for MNIST digit recognition without relying on high-level frameworks like TensorFlow or PyTorch

## Setup Instructions

### Prerequisites
- Python 3.13.7 
- Git (for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/holmen1/mnist-from-scratch.git
cd mnist-from-scratch
```

### 2. Set Up Virtual Environment
Create and activate a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# On Windows, use:
# venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- NumPy: For numerical computations and array operations
- pytest: For running unit tests

### 4. Set Up Testing Framework
This project uses pytest for unit testing. After installation, you can run tests with:

```bash
pytest
```

To run tests in a specific directory:

```bash
pytest tests/
```

Add test files in the `tests/` directory as you implement features.

## Project Structure
- `data/`: Data loading and preprocessing
- `models/`: Neural network implementation
- `utils/`: Helper functions (activations, losses)
- `scripts/`: Training and evaluation scripts
- `tests/`: Unit tests

## Next Steps
1. Implement data loading in `data/mnist_loader.py`
2. Build the neural network in `models/neural_network.py`
3. Add activation and loss functions in `utils/`
4. Train the model using `scripts/train.py`

