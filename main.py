from qiskit.algorithms import NumPyMinimumEigensolver, MinimumEigensolverResult
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms import VQE
from qiskit_aer.noise import NoiseModel
from qiskit_aer import Aer
import qiskit_ibm_runtime

import numpy as np
from pathlib import Path
import argparse
import pickle
import os
import warnings

from Utils import (Log, mitigation_meta, get_pauli_op, get_service, virtualize_single_results)
from mitigation_method import main as target_program

warnings.simplefilter("ignore")
pauli_str_dir = 'pauli_strings'
data_pkl_path = 'data.pkl'
setup_pkl_path = 'setup.pkl'



def load_pickle_files(args):
    """Loads result and configuration pickle files and processes them."""
    with open(data_pkl_path, 'rb') as data_file, open(setup_pkl_path, 'rb') as setup_file:
        loaded_result = pickle.load(data_file)
        loaded_vqe_inputs = pickle.load(setup_file)

    optimizer_name = loaded_vqe_inputs['optimizer']    
    output_path = loaded_vqe_inputs['output_dir']
    Log.info(f'Loaded the pickle files {data_pkl_path} and {setup_pkl_path}')

    if optimizer_name in ['DISQ', 'SPSA']:
        virtualize_single_results(args, loaded_result, loaded_vqe_inputs)
    Log.warning(f'Validation and Virtualization Result DONE')
    Log.warning(f'Output dir: {output_path}')


def preparation(args, service):
    """ Prepares the runtime environment by setting up necessary files and paths. """
    program_id = None
    upload_flag = False

    if args.upload > 0:
        upload_flag = True
        if args.optimizer in ['DISQ', 'SPSA']:
            Log.warning("\n" + "=" * 47 + f"\nRun {args.optimizer} Mitigation\n" + "=" * 47)
            file_name = "mitigation_method.py"
            script_path = os.path.join(os.getcwd(), file_name)
            metadata = mitigation_meta
            program_id = 'transient-noise-mitigation-JWjoXe5XgM'    # replace with yours

            if args.upload > 1:
                try:
                    service.update_program(program_id=program_id, data=script_path, metadata=metadata)
                except Exception as e:
                    Log.error(f'Error updating program: {str(e)}')
                    try:
                        program_id = service.upload_program(data=script_path, metadata=metadata)
                        service.pprint_programs()
                    except qiskit_ibm_runtime.exceptions.IBMNotAuthorizedError as nae:
                        Log.error('Failed to upload program - not authorized: ' + str(nae))
                        upload_flag = False
                    except Exception as upload_error:
                        Log.error('An unexpected error occurred while uploading the program: ' + str(upload_error))
                        upload_flag = False
                Log.warning(f'Run Program: {program_id}, file name: {file_name}')
        else:
            raise ValueError(f"Optimizer {args.optimizer} is not supported")

    """Sets up the output directory and initializes log files."""
    cwd = os.getcwd()
    result_dir = 'result'
    op_name = f"{args.optimizer}_"
    backend_tag = 'sim' if args.backend == 'ibmq_qasm_simulator' else args.backend
    outdir_name = f'{op_name}{args.output}_{backend_tag}'

    output_path = Path(cwd) / result_dir / outdir_name
    output_path.mkdir(parents=True, exist_ok=True)

    os.chdir(output_path)
    log_file_name = f'{args.output}_debug.log'
    log_path = output_path / log_file_name
    if log_path.exists():
        log_path.unlink()
    Log.add(str(log_file_name))
    Log.debug(args)
    Log.debug(f'Current directory: {output_path}')

    return upload_flag, program_id, output_path




def main(args, service):
    upload_flag, program_id, output_path = preparation(args, service)

    ''' Get the qubit operator '''
    qubit_op, problem = get_pauli_op(molecule_name=args.molecule)
    num_qubits = qubit_op.num_qubits

    np.random.seed(args.seed)

    if args.ansatz == "SU2":
        var_form = EfficientSU2(num_qubits, reps=args.rep, entanglement="full")
    elif args.ansatz == "RA":
        var_form = RealAmplitudes(num_qubits, reps=args.rep, entanglement="full")
    initial_point = np.full(var_form.num_parameters, args.para*np.pi)
 
    if isinstance(problem, float):
        Log.warning(f'Energy: {problem}')
    elif problem != None:
        npme = NumPyMinimumEigensolver()
        sol = MinimumEigensolverResult()
        ref_result = npme.compute_minimum_eigenvalue(operator=qubit_op)
        sol.eigenvalue = ref_result.eigenvalue
        real_solution = problem.interpret(sol).total_energies[0]
        Log.warning(f'Reference eigenvalue: {sol.eigenvalue.real:.5f}')
        Log.warning(f'Energy: {real_solution.real}')

    ''' Execute the program '''
    backend = service.get_backend(args.backend)
    vqe_inputs = {
        "ansatz": var_form,
        "operator": qubit_op,
        "initial_point": initial_point,
        "shots": 1024,
        "iteration": args.iter,
        "output_dir": str(output_path),
        "seed": args.seed,
        "optimizer": args.optimizer,
        "factor": args.factor,
        "c": args.cal_c,
        "alpha": args.cal_a,
        "prime_th": args.prime_th,
        "threshold": args.threshold,
        "method": args.method,
    }
    result = None
    if upload_flag:
        backend = service.get_backend(args.backend)
        Log.warning(f"Program: <{program_id}>\tBackend: <{backend.name}>")
        job = service.run(program_id=program_id, inputs=vqe_inputs, options={"backend_name": backend.name})
        Log.info(f"Job ID: {job.job_id()}")
        print(job.status())
        result = job.result()
    else:
        backend = Aer.get_backend('aer_simulator')
        result = target_program(backend, None, **vqe_inputs)

    ''' Result processing '''
    with open(setup_pkl_path, 'wb') as outp:
        vqe_inputs["problem"] = problem
        pickle.dump(vqe_inputs, outp, pickle.HIGHEST_PROTOCOL)

    with open(data_pkl_path, 'wb') as outp:
        pickle.dump(result, outp, pickle.HIGHEST_PROTOCOL)

    Log.info(f'validate the pkl files {data_pkl_path} and {setup_pkl_path} with virtualization')
    load_pickle_files(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic arguments
    parser.add_argument('--backend', default='ibmq_qasm_simulator', help='choose the backend')
    parser.add_argument('--iter', type=int, default=30, help='number of iteration')
    parser.add_argument('--output', default='temp', help='output directory')
    parser.add_argument('--seed', type=int, default=None, help='set seeds')
    parser.add_argument('--service', default="ncstate", choices=["ncstate", "MIT"], help='chose the service')
    parser.add_argument('--check', type=int, default=None, help='check avaiable device')

    parser.add_argument('--molecule', default='LiH_1-4', help='set molecule')
    parser.add_argument('--ansatz', default="RA", choices=['RA', 'SU2'], help='set ansatz')
    parser.add_argument('--rep',  type=int, default=6, help='set ansatz repetition')
    parser.add_argument('--para',  type=float, default=1, help='set ansatz initial parameter')

    # optimizer arguments
    parser.add_argument('--optimizer', default='DISQ', choices=['DISQ', 'SPSA'], help='Disq or SPSA optimizers')
    parser.add_argument('--method', default='amp_p', choices=['amp_a', 'amp_p', 'grad'], 
                        help='amplitude method or gradient method')
    parser.add_argument('--cal_c', type=float, default=0.2, help='set c value in calibrate')
    parser.add_argument('--cal_a', type=float, default=0.9, help='set alpha value in calibrate')

    # transient noise arguments
    parser.add_argument('--factor', type=float, default=0, help='set transient noise factor')
    parser.add_argument('--threshold', type=float, default=0.9, help='set transient noise threshold')
    parser.add_argument('--prime_th', type=float, default=0.8, help='set prime threshold')
    parser.add_argument('--upload', type=int, default=2, choices=[0, 1, 2], 
                        help='upload the script to runtime, 0:not upload, 1:run exist program, 2: create new program')
    
    args = parser.parse_args()

    ''' Execute the scripts '''
    if args.check is not None:
        service = get_service(args.service, args.check)
    else:
        service = get_service(args.service)
        main(args, service)
