# DISQ: Dynamic Iteration Skipping for Variational Quantum Algorithms

This repository contains the implementation of the Variational Quantum Eigensolver (VQE) algorithm with noise mitigation methods. It is designed to run on IBM Quantum hardware through the Qiskit runtime service but also includes a simulation mode for local testing and development.

## Project Structure

- `main.py`: The main script to setup, run, and process the VQE computations.
- `mitigation_method.py`: Contains specific implementations for error mitigation techniques.
- `Utils.py`: Helper functions and utilities for logging, data loading, and service interaction.
- `requirements.txt`: Python dependencies required for the project.
- `pauli_strings/`: Directory containing Pauli operators in string form.
- `result/`: Directory where the outputs and intermediate results will be saved.

## Setup

### Prerequisites

- Python 3.7+
- pip
- Virtual environment (recommended)

### Environment Setup

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv_name
source venv_name/bin/activate
```

Install the required Python modules:

```bash
pip install -r requirements.txt
```

### Configuration

Before running the `main.py` script, ensure that you have set up your Qiskit runtime service correctly and that you have the necessary credentials configured in your environment (refer to [Qiskit documentation](https://qiskit.org/documentation/) for more details).

## Usage

To run the main VQE program, use the following command from the terminal:

```bash
python main.py --backend ibmq_qasm_simulator --optimizer DISQ
```

### Command Line Arguments

- `--backend`: The quantum backend to use (default: `ibmq_qasm_simulator`).
- `--iter`: Number of iterations for the optimizer (default: 20).
- `--output`: Name for the output directory (default: `temp`).
- `--seed`: Random seed for reproducibility (optional).
- `--ansatz`: The type of ansatz to use (`SU2` or `RA`, default: `RA`).
- `--rep`: Number of repetitions for the ansatz (default: 6).
- `--optimizer`: The optimizer to use (`DISQ` or `SPSA`, default: `DISQ`).
- `--upload`: Flag to upload the script to the runtime (0: no upload, 1: run existing program, 2: create new program, default: 2).

For a full list of options, use the `-h` or `--help` flag when running the script.

## Output

Results and logs will be stored in the `result/` directory specified by the `--output` flag. This includes intermediate computational results and final outcomes.


## Contact
Junyao Zhang [Email](mailto:jz420@duke.edu), [Github issue](https://github.com/JJJayyyy/Disq/issues)


## Publications

- Zhang, Junyao, Hanrui Wang, Gokul Subramanian Ravi, Frederic T. Chong, Song Han, Frank Mueller, and Yiran Chen. "Disq: Dynamic iteration skipping for variational quantum algorithms." In 2023 IEEE International Conference on Quantum Computing and Engineering (QCE), vol. 1, pp. 1062-1073. IEEE, 2023.
  ([Link](https://ieeexplore.ieee.org/abstract/document/10313742/?casa_token=lKKp9qOtwDwAAAAA:y4rzb5cYf-Xwwt4y08EPz6S4bvOcnwOjLrxKRNRIF4nVxwVqpJg5sV4j5DLI86ttxnIifwUzSw))

## Citation
```
@inproceedings{zhang2023disq,
  title={Disq: Dynamic iteration skipping for variational quantum algorithms},
  author={Zhang, Junyao and Wang, Hanrui and Ravi, Gokul Subramanian and Chong, Frederic T and Han, Song and Mueller, Frank and Chen, Yiran},
  booktitle={2023 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  volume={1},
  pages={1062--1073},
  year={2023},
  organization={IEEE}
}
```