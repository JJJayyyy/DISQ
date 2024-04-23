from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.opflow import PauliSumOp
from qiskit.algorithms import NumPyMinimumEigensolver, MinimumEigensolverResult
from qiskit.providers.ibmq import least_busy
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import IBMQ

import matplotlib.pyplot as plt
from loguru import logger as Log
from pathlib import Path
import numpy as np
import warnings
import sys
import re
import os


warnings.simplefilter("ignore")

Log.remove()
Log.add(sys.stderr,
        level='DEBUG',
        format=('<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> '
                '<level>{message}</level>'))


pauli_str_path = os.path.join(os.getcwd(), "pauli_strings")


mitigation_meta = {
    "name": "transient_noise_mitigation",
    "description": "VQE with customized transient noise mitigation algorithm.",
    "is_public": False,
    "version": 1.0,
    "max_execution_time": 100000,
    "spec": {}
}

mitigation_meta["spec"]["parameters"] = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "properties": {
        "operator": {
            "description": "Hamiltonian whose ground state we want to find.", 
            "type": "PauliSumOp"
        },
        "ansatz": {
            "description": "Name of ansatz quantum circuit to use, default='EfficientSU2'",
            "type": "QuantumCircuit",
            "default": "EfficientSU2"
        },
        "iteration": {
            "description": "iteration",
            "type": "integer"
        },
        "seed": {
            "description": "seed",
            "type": "integer",
        },
        "initial_parameters": {
            "description": "Initial vector of parameters. This is a numpy array.", 
            "type": "Union[numpy.ndarray, str]"
        },
        "shots": {
            "description": "The number of shots used for each circuit evaluation.",
            "type": "integer"
        },
        "noise_model": {
            "description": "noise model to use",
            "type": "object",
            "default": False
        },
        "optimizer": {
            "description": "optimizer for VQE",
            "type": "string",
            "default": False
        },
        "factor": {
            "description": "transient noise factor",
            "type": "float",
            "default": False
        }
    },
    "required": ["operator", "ansatz"]
}

mitigation_meta["spec"]["return_values"] = {
    "$schema": "https://json-schema.org/draft/2019-09/schema",
    "description": "Final result in dictionary format",
    "type": "object"
}




def get_service(name="", check_available=None):
    """
    Configure and return a Qiskit runtime service based on provided parameters.

    Args:
    name (str): Name identifier for the service. Uses a default token and instance if not provided.
    check_available (int, optional): Number of qubits to check for availability on devices.

    Returns:
    QiskitRuntimeService: Configured runtime service object.
    """
    # Custom configuration if name is provided
    if name == "":
        token = "your_token_here"
        instance = 'instance_here'
        hub = "hub_name_here"
    else:
        Log.error("Invalid token provided")

    # Initialize the runtime service
    service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
    Log.info(f"Current service initialized: {name}")

    # Check for available devices if requested
    if check_available is not None:
        IBMQ.save_account(token=token, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=hub)

        devices = provider.backends(filters=lambda x: x.configuration().n_qubits >= check_available
                                    and not x.configuration().simulator)
        if devices:
            least_busy_device = least_busy(devices)
            Log.info(f"Least busy device with at least {check_available} qubits: {least_busy_device}")
        else:
            Log.info("No available devices found that meet the criteria.")

    return service



""" Pauli String """

def load_pauli_string(file_path):
    """
    Load and parse a Pauli string from a file.

    Args:
        file_path (str): Path to the file containing the Pauli string data.

    Returns:
        tuple: A tuple containing:
               - PauliSumOp: Quantum Operator represented by the Pauli strings.
               - float: The result value extracted from the file.
    """
    regex_patterns = {
        'result': re.compile(r'FCI:\s+(-?\d*\.?\d*)\s*'),
        'pauli_str': re.compile(r'(-?\d*\.?\d*e?-?\d*)\s+\[(.*)\]\s+')
    }
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        file_data = {'pauli_str': []}
        for line in lines:
            for key, regex in regex_patterns.items():
                match = regex.match(line)
                if match:
                    if key == 'result':
                        file_data[key] = float(match.group(1))
                    else:
                        file_data[key].append({
                            'coefficient': float(match.group(1)), 
                            'operators': match.group(2)
                        })

        pauli_list = process_pauli_strings(file_data['pauli_str'])
        pauli_op = PauliSumOp.from_list(pauli_list)
        
        Log.info(f"Loaded Pauli strings from {file_path} with result {file_data['result']}")
        return pauli_op, file_data['result']

    except FileNotFoundError:
        Log.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        Log.error(f"An error occurred: {e}")
        raise


def process_pauli_strings(pauli_data):
    """
    Convert raw Pauli string data to formatted PauliSumOp input.

    Args:
        pauli_data (list): List of dictionaries containing coefficients and operator strings.

    Returns:
        list: Processed list of tuples for PauliSumOp initialization.
    """
    pauli_list = []
    for data in pauli_data:
        operators = data['operators'].split()
        pauli_str = ''
        last_idx = 0

        for operator in operators:
            index = int(operator[1])
            if index < last_idx:
                Log.error("Operator indices are not in increasing order.")
                raise ValueError("Operator indices should be in strictly increasing order.")
            pauli_str += 'I' * (index - last_idx) + operator[0]
            last_idx = index + 1

        if last_idx < 6:
            pauli_str += 'I' * (6 - last_idx)
        
        if len(pauli_str) != 6:
            Log.error("Invalid Pauli string length.")
            raise ValueError("Generated Pauli string does not have the correct length.")
        
        pauli_list.append((pauli_str, data['coefficient']))
    
    return pauli_list




def get_pauli_op(molecule_name='H2', reduce=False):
    if molecule_name == 'H2':
        ml = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], 
                                ["H", [0.0, 0.0, 1.5]]], 
                                multiplicity=1, charge=0)
        driver = ElectronicStructureMoleculeDriver(molecule=ml, basis="sto3g",
            driver_type=ElectronicStructureDriverType.PYSCF)
        properties = driver.run()
        problem = ElectronicStructureProblem(driver)
        
    elif molecule_name == 'HeH':
        ml = Molecule(geometry=[["He", [0.0, 0.0, -0.87818361]], 
                                ["H", [0.0, 0.0, 0.87818361]]], 
                                multiplicity=1, charge=1)
        driver = ElectronicStructureMoleculeDriver(molecule=ml, basis="sto3g",
            driver_type=ElectronicStructureDriverType.PYSCF)
        properties = driver.run()
        problem = ElectronicStructureProblem(driver)

    elif molecule_name == 'LiH':
        ml = Molecule(geometry=[["Li", [0.0, 0.0, 0.0]], 
                               ["H", [1.5, 0.0, 0]]], 
                               multiplicity=1, charge=0)
        driver = ElectronicStructureMoleculeDriver(molecule=ml, basis="sto3g", 
            driver_type=ElectronicStructureDriverType.PYSCF)
        problem = ElectronicStructureProblem(driver, 
        [FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[4, 5])])
        reduce = True

    elif molecule_name == 'HF':
        ml = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], 
                               ["F", [1.5, 0.0, 0]],],
                               multiplicity=1, charge=0)
        driver = ElectronicStructureMoleculeDriver(molecule=ml, basis="sto3g", 
            driver_type=ElectronicStructureDriverType.PYSCF)
        problem = ElectronicStructureProblem(driver, [FreezeCoreTransformer(freeze_core=True, 
            remove_orbitals=[4,5])])
        reduce = True

    elif molecule_name == 'NaH':
        ml = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], 
                               ["Na", [1.5, 0.0, 0]],],
                               multiplicity=1, charge=0)
        driver = ElectronicStructureMoleculeDriver(molecule=ml, basis="sto3g", 
            driver_type=ElectronicStructureDriverType.PYSCF)
        problem = ElectronicStructureProblem(driver, 
        [FreezeCoreTransformer(freeze_core=True, remove_orbitals=[4, 5])])
        reduce = True

    else:
        pauli_op, value = load_pauli_string(Path(pauli_str_path).joinpath(f'{molecule_name}.txt'))
        return pauli_op, value
   
    second_q_ops = problem.second_q_ops()   # -> FermionicOp (list)
    hamiltonian = second_q_ops[0]           # get the first element, list_size=1, (list)
    # hamiltonian = second_q_ops['ElectronicEnergy']
    num_particles = problem.num_particles
    print(f'Problem var-#spin_orb: {problem.num_spin_orbitals}\t#particles: {num_particles}')
    # same energy but latter has less qubit 
    if reduce:
        converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
        qubit_op = converter.convert(hamiltonian, num_particles=problem.num_particles)
    else:
        converter = QubitConverter(ParityMapper())
        qubit_op = converter.convert(hamiltonian)
    print(f'qubit_op: #qubits = {qubit_op.num_qubits}\t#len = {len(qubit_op)}\tsample = {qubit_op[0]}\n{type(qubit_op)}')
    return qubit_op, problem



       



""" Virtualization """

def draw_energy_image(energy_list, label_list, file_name, real_solution=None, noise_history=None):
    plt.figure(figsize = (10, 5))
    if len(label_list) > 0:
        for values, name in zip(energy_list, label_list):
            plt.plot(range(1, len(values)+1), values, label=f"{name}")
    else:
        for values in energy_list:
            plt.plot(range(1, len(values)+1), values)

    if real_solution is not None:
        plt.axhline(y=real_solution, color="tab:red", ls="--", label="Target")

    if noise_history is not None:
        gap = max(values) - min(values)
        plt.vlines(x=noise_history, ymin = min(values)-gap*0.1, ymax = max(values)+gap*0.1, 
          color="tab:gray", ls=":", label="TNoise")

    plt.xlabel('Eval count')
    plt.ylabel('Energy')
    if len(label_list) > 0:
        plt.legend()
    plt.savefig(file_name)


def virtualize_single_results(args, result, vqe_inputs):
    optimizer = vqe_inputs['optimizer']
    problem = vqe_inputs['problem']
    labels = [optimizer]

    if isinstance(problem, float):
        ref_v = problem
    elif problem != None:
        npme = NumPyMinimumEigensolver()
        sol = MinimumEigensolverResult()
        ref_result = npme.compute_minimum_eigenvalue(operator=vqe_inputs['operator'])
        sol.eigenvalue = ref_result.eigenvalue
        ref_v = sol.eigenvalue.real
        ref_e = problem.interpret(sol).total_energies[0]
            
    values     = result["optimizer_history"]["values"]
    avg_values = result["optimizer_history"]["values_a"]
    all_values = result["optimizer_history"]["values_all"]
    p_values   = result["optimizer_history"]["sum_args_list"]
    ap_values  = result["optimizer_history"]["abs_args_list"]
    avg_p_values = np.mean(p_values, axis=1)
    avg_ap_values = np.mean(ap_values, axis=1)

    nfun_ev = result["other_data"]["nfun_ev"]
    prime_idx = result["other_data"]["prime_idx"]
    expect_op_num = result["other_data"]["expect_op_num"]
    noise_history = result["other_data"]["noise_history"]
    mitigation_history = result["other_data"]["mitigation_history"]
    execution_time_history = result["other_data"]["execution_time_history"]
    actual_time_his = result["other_data"]["actual_execution_time_history"]
    re = result["result_list"]

    ''' eigenvalue '''
    Log.debug('Prime threshold: {}\tNum of op: {}\tPrime op idx: {}'
        .format(args.prime_th, expect_op_num, prime_idx))
    Log.debug(f'Total noises num: {len(noise_history)}')
    missed_noise = np.setdiff1d(noise_history, mitigation_history)
    Log.debug(f'Missed noise ({(len(missed_noise))}): {missed_noise}')
    overmiti = np.setdiff1d(mitigation_history, noise_history)
    Log.debug(f'Over mitigations ({len(overmiti)}): {overmiti}')
    Log.debug(f'Total executions: {nfun_ev}')

    Log.debug('Avg execution time ({}/{}): {}'.format(len(execution_time_history), 
        args.iter, np.sum(execution_time_history)/args.iter))
    Log.debug('Actual execution time ({}/{}): {}'.format(len(actual_time_his), 
        args.iter, np.sum(actual_time_his)/len(actual_time_his)))

    for i in range(len(re)):
        Log.debug(f'{labels[i]} Noise: {re[i]:<10.5f}\tDelta: {(re[i]-ref_v):.5f}')

    if args.backend != 'ibmq_qasm_simulator':
        noise_history = None
    draw_energy_image([values], labels, f'{args.output}.png', ref_v)
    draw_energy_image([avg_values], labels, f'{args.output}_avg.png', ref_v, noise_history=noise_history)
    draw_energy_image(all_values, labels, f'{args.output}_all.png', ref_v)
    draw_energy_image([avg_values, avg_p_values], [labels[0], 'prime'], f'{args.output}_p.png', ref_v)
    draw_energy_image([avg_values, avg_p_values, avg_ap_values], [labels[0], 'prime', 'abs_prime'], f'{args.output}_ap.png', ref_v)
        
    ''' energy '''
    if problem != None and not isinstance(problem, float):
        for i in range(len(re)):
            s = MinimumEigensolverResult()
            s.eigenvalue = re[i]
            energy = problem.interpret(s).total_energies[0].real
            Log.debug(f'{labels[i]} Noise Energy: {energy:<10.5f}\tDelta: {(energy-ref_e.real):.5f}')

        values_energy = [[] for _ in range(len(values))]

        for j in range(len(values)):
            sol = MinimumEigensolverResult()
            sol.eigenvalue = values[j]
            sol = problem.interpret(sol).total_energies[0]
            values_energy[0].append(sol)
        draw_energy_image(values_energy, labels, f'{args.output}_in_energy.png', ref_e.real)

        all_values_energy = [[] for _ in range(len(all_values))]
        for i in range(len(all_values)):
            for j in range(len(all_values[i])):
                sol = MinimumEigensolverResult()
                sol.eigenvalue = all_values[i][j]
                sol = problem.interpret(sol).total_energies[0]
                all_values_energy[i].append(sol)
        draw_energy_image(all_values_energy, labels, f'{args.output}_all_in_energy.png', ref_e.real)