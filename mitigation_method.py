from __future__ import annotations
import copy
import logging
import warnings
import numpy as np
import math
from time import time
from functools import partial
from qiskit import QiskitError
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import (CircuitSampler, CircuitStateFn, DictStateFn, ExpectationBase,
    ExpectationFactory, ListOp, SummedOp, ComposedOp, OperatorBase, StateFn)
from qiskit.opflow.gradients import GradientBase
from qiskit.opflow.converters.circuit_sampler import _filter_params, OperatorCache
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.providers import Backend
from qiskit.algorithms import VQEResult
from qiskit.algorithms.exceptions import AlgorithmError
from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.algorithms.optimizers import Minimizer, Optimizer
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import MinimumEigensolver
from qiskit.algorithms.optimizers.optimizer import OptimizerSupportLevel, OptimizerResult, POINT
from qiskit.algorithms.minimum_eigen_solvers.vqe import _validate_initial_point
from typing import Iterator, Optional, Union, Callable, Tuple, Dict, List, Any, cast
from qiskit import IBMQ
from collections import deque

# number of function evaluations, parameters, loss, stepsize, accepted
CALLBACK = Callable[[int, np.ndarray, float, float, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, float, bool], bool]
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore")

REF_LIST=[-1, -2, -3]
NOISE_ENABLE=True



def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
    indices = algorithm_globals.random.choice(list(range(dim)), size=perturbation_dims, replace=False)
    result = np.zeros(dim)
    result[indices] = pert
    return result


def powerseries(eta=0.01, power=2, offset=0):
    """Yield a series decreasing by a powerlaw."""
    n = 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1


def constant(eta=0.01):
    """Yield a constant series."""
    while True:
        yield eta


def _batch_evaluate(
        function, 
        points,
        max_evals_grouped, 
        unpack_points=False, 
        calibrate=False, 
        para_op_pair=None, 
        indicator_input=None
    ):
    """Evaluate a function on all points with batches of max_evals_grouped.

    The points are a list of inputs, as ``[in1, in2, in3, ...]``. If the individual
    inputs are tuples (because the function takes multiple inputs), set ``unpack_points`` to ``True``.
    """

    # if the function cannot handle lists of points as input, cover this case immediately
    if max_evals_grouped is None or max_evals_grouped == 1:
        # support functions with multiple arguments where the points are given in a tuple
        return function(points, calibrate, para_op_pair, indicator_input)
    num_points = len(points)

    # get the number of batches
    num_batches = num_points // max_evals_grouped
    if num_points % max_evals_grouped != 0:
        num_batches += 1

    # split the points
    batched_points = np.array_split(np.asarray(points), num_batches)
    results = []
    for batch in batched_points:
        if unpack_points:
            batch = _repack_points(batch)
            results += _as_list(function(*batch))
        else:
            results += _as_list(function(batch))
    return results


def _as_list(obj):
    """Convert a list or numpy array into a list."""
    return obj.tolist() if isinstance(obj, np.ndarray) else obj


def _repack_points(points):
    """Turn a list of tuples of points into a tuple of lists of points.
    E.g. turns
        [(a1, a2, a3), (b1, b2, b3)]
    into
        ([a1, b1], [a2, b2], [a3, b3])
    where all elements are np.ndarray.
    """
    num_sets = len(points[0])  # length of (a1, a2, a3)
    return ([x[i] for x in points] for i in range(num_sets))


def _make_spd(matrix, bias=0.01):
    identity = np.identity(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return psd + bias * identity


def _validate_pert_and_learningrate(perturbation, learning_rate):
    if learning_rate is None or perturbation is None:
        raise ValueError("If one of learning rate or perturbation is set, both must be set.")

    if isinstance(perturbation, float):
        def get_eps():
            return constant(perturbation)
    elif isinstance(perturbation, (list, np.ndarray)):
        def get_eps():
            return iter(perturbation)
    else:
        get_eps = perturbation

    if isinstance(learning_rate, float):
        def get_eta():
            return constant(learning_rate)
    elif isinstance(learning_rate, (list, np.ndarray)):
        def get_eta():
            return iter(learning_rate)
    else:
        get_eta = learning_rate

    return get_eta, get_eps



def mod_eval_with_th(oplist, prime_th, calculate_prime=True, print_level=0) -> Union[OperatorBase, complex]:
    if not isinstance(oplist, ListOp):
        raise NotImplementedError("It is not a ListOp")

    sumop_energys = []
    prime_arg_list = []
    assert oplist.coeff == 1
    # ListOp[SummedOp[ComposedOp,] ComposedOp], 
    # #Op>1 => summedOp, #Op==1 => ComposedOp
    for idx, sum_op_list in enumerate(oplist):
        if isinstance(sum_op_list, SummedOp):
            evals = [op.eval() for op in sum_op_list]
        elif isinstance(sum_op_list, ComposedOp):
            evals = [sum_op_list.eval()]
        else:
            raise RuntimeError(f"Encounter {type(sum_op_list)} is not in SummedOp or ComposedOp")

        op_energy = oplist.coeff * sum_op_list.coeff * np.sum(evals, axis=0)
        evals = oplist.coeff * sum_op_list.coeff * np.array(evals)
        evals_sum = np.sum(evals)
        assert evals_sum == op_energy
        sumop_energys.append(evals_sum)

        if calculate_prime:
            evals = np.real(evals)
            abs_sum = np.sum(np.abs(evals))
            _sum = np.sum(evals)
            eval_percent = 0
            prime_dict = {}
            indexed_list = [(j, x) for j, x in enumerate(evals)]
            sorted_items = sorted(indexed_list, key=lambda x: x[1])
            for j, v in sorted_items:
                eval_percent += np.abs(v/_sum)
                prime_dict[j] = v
                if eval_percent > prime_th:
                    break

            prime_arg_list.append(prime_dict)

            if print_level > 0:
                print('eval:', evals)
                print(f'sorted_items : {sorted_items}')
                print(f'abs_sim : {abs_sum}, evals_sum: {evals_sum}')
                print(f'{idx} : {prime_dict}\n')
        
    listop_energy = np.real(sumop_energys)
    if calculate_prime:
        return listop_energy, np.array(prime_arg_list)
    else:
        return listop_energy


def mod_eval_with_ob(oplist, op_idx, calculate_prime=True, print_level=0) -> Union[OperatorBase, complex]:
    if not isinstance(oplist, ListOp):
        raise NotImplementedError("It is not a ListOp")

    sumop_energys = []
    prime_arg_list = []
    assert oplist.coeff == 1
    # ListOp[SummedOp[ComposedOp,] ComposedOp], 
    # #Op>1 => summedOp, #Op==1 => ComposedOp
    for idx, sum_op_list in enumerate(oplist):
        if isinstance(sum_op_list, SummedOp):
            evals = [op.eval() for op in sum_op_list]
        elif isinstance(sum_op_list, ComposedOp):
            evals = [sum_op_list.eval()]
        else:
            evals = None
            raise RuntimeError(f"Encounter {type(sum_op_list)} is not in SummedOp or ComposedOp")

        op_energy = oplist.coeff * sum_op_list.coeff * np.sum(evals, axis=0)
        evals = oplist.coeff * sum_op_list.coeff * np.array(evals)
        evals_sum = np.sum(evals)
        assert evals_sum == op_energy
        sumop_energys.append(evals_sum)

        if calculate_prime:
            evals = np.real(evals)
            prime_dict = {}
            for j in op_idx:
                prime_dict[j] = evals[j]
            prime_arg_list.append(prime_dict)

            if print_level > 0:
                print('eval:', evals)
                print(f'{idx} : {prime_dict}\n')
        
    listop_energy = np.real(sumop_energys)
    if calculate_prime:
        return listop_energy, np.array(prime_arg_list)
    else:
        return listop_energy


def mod_eval_exp_step(oplist, op_idx, print_level=0) -> Union[OperatorBase, complex]:
    if not isinstance(oplist, ListOp):
        raise NotImplementedError("It is not a ListOp")

    sumop_energys = []
    prime_arg_list = []
    assert oplist.coeff == 1
    for idx, sum_op_list in enumerate(oplist):
        if isinstance(sum_op_list, SummedOp):
            evals = [op.eval() for op in sum_op_list]
        elif isinstance(sum_op_list, ComposedOp):
            evals = [sum_op_list.eval()]
        else:
            raise RuntimeError(f"Encounter {type(sum_op_list)} is not in SummedOp or ComposedOp")
        op_energy = oplist.coeff * sum_op_list.coeff * np.sum(evals, axis=0)
        evals = oplist.coeff * sum_op_list.coeff * np.array(evals)
        evals_sum = np.sum(evals)
        assert evals_sum == op_energy
        sumop_energys.append(evals_sum)

        evals = np.real(evals)
        prime_dict = {}
        for idx, item in zip(op_idx, evals):
            prime_dict[idx] = item
        prime_arg_list.append(prime_dict)

        if print_level > 0:
            print('eval:', evals)
            print(f'{idx} : {prime_dict}\n')
        
    listop_energy = np.real(sumop_energys)
    return listop_energy, np.array(prime_arg_list)


def mod_eval(oplist, print_level=0) -> Union[OperatorBase, complex]:
    if not isinstance(oplist, ListOp):
        raise NotImplementedError("It is not a ListOp")

    sumop_energys = []
    max_arg_list = []
    assert oplist.coeff == 1

    # ListOp[SummedOp[ComposedOp, ...], ComposedOp], 
    # #Op>1 => summedOp, #Op==1 => ComposedOp
    for i, sum_op_list in enumerate(oplist):
        if isinstance(sum_op_list, SummedOp):
            evals = [op.eval() for op in sum_op_list]
        elif isinstance(sum_op_list, ComposedOp):
            evals = [sum_op_list.eval()]
        else:
            evals = None
            raise RuntimeError(f"Encounter {type(sum_op_list)} is not in SummedOp or ComposedOp")

        assert sum_op_list.coeff == 1
        op_energy = oplist.coeff * sum_op_list.coeff * np.sum(evals, axis=0)
        sumop_energys.append(op_energy)

        idx = np.argmax(np.abs(evals))
        max_part = np.real(evals[idx])
        max_arg_list.append({idx : max_part})

        if print_level > 0:
            print(f'\nPara < {i} > (#op {len(evals)}): {evals}')
            if print_level > 1:
                print('Max pair: {} -> {:<30} Minor prop: {}'
                .format(idx, evals[idx], abs((max_part-op_energy) / op_energy)))
    listop_energy = np.real(sumop_energys)
    return listop_energy, np.array(max_arg_list)


def mod_sample_circuits(
    sampler,
    circuit_sfns: Optional[List[CircuitStateFn]] = None,
    param_bindings: Optional[List[Dict[Parameter, float]]] = None,
    para_op_pair: dict = None,
) -> Dict[int, List[StateFn]]:
    if not circuit_sfns and not sampler._transpiled_circ_cache:
        raise ("CircuitStateFn is empty and there is no cache.")

    if circuit_sfns:
        sampler._transpiled_circ_templates = None
        if sampler._statevector or circuit_sfns[0].from_operator:
            circuits = [op_c.to_circuit(meas=False) for op_c in circuit_sfns]
        else:
            circuits = [op_c.to_circuit(meas=True) for op_c in circuit_sfns]

        try:
            sampler._transpiled_circ_cache = sampler.quantum_instance.transpile(
                circuits, pass_manager=sampler.quantum_instance.unbound_pass_manager)
        except QiskitError:
            logger.debug(
                r"CircuitSampler failed to transpile circuits with unbound "
                r"parameters. Attempting to transpile only when circuits are bound "
                r"now, but this can hurt performance due to repeated transpilation.")
            sampler._transpile_before_bind = False
            sampler._transpiled_circ_cache = circuits
    else:
        circuit_sfns = list(sampler._circuit_ops_cache.values())

    if param_bindings is not None:
        if sampler._param_qobj:
            start_time = time()
            ready_circs = sampler._prepare_parameterized_run_config(param_bindings)
            end_time = time()
            logger.info("Parameter conversion %.5f (ms)", (end_time - start_time) * 1000)
        else:
            start_time = time()
            ready_circs = []
            for i, circ in enumerate(sampler._transpiled_circ_cache):
                for j, binding in enumerate(param_bindings):
                    if para_op_pair:
                        if j in para_op_pair.keys() and i not in para_op_pair[j]:
                            continue
                    ready_circs.append(circ.assign_parameters(_filter_params(circ, binding)))
                        
            end_time = time()
            logger.info('read_circ lens: {}, Parameter binding {:.5f} (ms)'
                    .format(len(ready_circs), (end_time - start_time) * 1000))
    else:
        ready_circs = sampler._transpiled_circ_cache

    # run transpiler passes on bound circuits
    if sampler._transpile_before_bind and sampler.quantum_instance.bound_pass_manager is not None:
        ready_circs = sampler.quantum_instance.transpile(
            ready_circs, pass_manager=sampler.quantum_instance.bound_pass_manager)

    results = sampler.quantum_instance.execute(ready_circs, had_transpiled=sampler._transpile_before_bind)

    if param_bindings is not None and sampler._param_qobj:
        sampler._clean_parameterized_run_config()

    sampled_statefn_dicts = {}
    circ_index = 0
    
    for i, op_c in enumerate(circuit_sfns):
        reps = len(param_bindings) if param_bindings is not None else 1
        c_statefns = []
        for j in range(reps):
            if para_op_pair:
                # if j in para_op_pair.keys() and para_op_pair[j] != i:
                if j in para_op_pair.keys() and i not in para_op_pair[j]:
                    continue
            circ_results = results.data(circ_index)

            if "expval_measurement" in circ_results:
                avg = circ_results["expval_measurement"]
                # Will be replaced with just avg when eval is called later
                num_qubits = circuit_sfns[0].num_qubits
                result_sfn = DictStateFn("0" * num_qubits,
                                         coeff=avg * op_c.coeff,
                                         is_measurement=op_c.is_measurement,
                                         from_operator=op_c.from_operator,)
            elif sampler._statevector:
                result_sfn = StateFn(op_c.coeff * results.get_statevector(circ_index),
                                     is_measurement=op_c.is_measurement,)
            else:
                shots = sampler.quantum_instance._run_config.shots
                result_sfn = DictStateFn({b: (v/shots) ** 0.5 * op_c.coeff
                                          for (b, v) in results.get_counts(circ_index).items()},
                                          is_measurement=op_c.is_measurement,
                                          from_operator=op_c.from_operator,)
            if sampler._attach_results:
                result_sfn.execution_results = circ_results
            c_statefns.append(result_sfn)
            circ_index += 1

        sampled_statefn_dicts[id(op_c)] = c_statefns

    return sampled_statefn_dicts


def modconvert(
    sampler: CircuitSampler,
    operator: OperatorBase,
    params: Optional[Dict[Parameter, Union[float, List[float], List[List[float]]]]] = None,
    para_op_pair: dict = None,
) -> OperatorBase: 
    op_id = operator.instance_id     # check if the operator should be cached
    if op_id not in sampler._cached_ops.keys():
        if sampler._caching == "last": # delete cache if we only want to cache one operator
            sampler.clear_cache()

        operator_dicts_replaced = operator.to_circuit_op()
        sampler._reduced_op_cache = operator_dicts_replaced.reduce()
        sampler._circuit_ops_cache = {}
        sampler._extract_circuitstatefns(sampler._reduced_op_cache)
        if not sampler._circuit_ops_cache:
            raise ("Circuits are empty. should be in type of CircuitStateFn or its ListOp.")

        sampler._transpiled_circ_cache = None
        sampler._transpile_before_bind = True
    else:
        # load the cached circuits
        sampler._reduced_op_cache = sampler._cached_ops[op_id].reduced_op_cache
        sampler._circuit_ops_cache = sampler._cached_ops[op_id].circuit_ops_cache
        sampler._transpiled_circ_cache = sampler._cached_ops[op_id].transpiled_circ_cache
        sampler._transpile_before_bind = sampler._cached_ops[op_id].transpile_before_bind
        sampler._transpiled_circ_templates = sampler._cached_ops[op_id].transpiled_circ_templates

    return_as_list = False
    if params is not None and len(params.keys()) > 0:
        p_0 = list(params.values())[0]
        if isinstance(p_0, (list, np.ndarray)):
            num_parameterizations = len(p_0)
            param_bindings = [{param: value_list[i] for param, value_list in params.items()}
                                for i in range(num_parameterizations)]
            return_as_list = True
        else:
            num_parameterizations = 1
            param_bindings = [params]
    else:
        param_bindings = None
        num_parameterizations = 1

    # Don't pass circuits if we have in the cache, the sampling function knows to use the cache
    circs = list(sampler._circuit_ops_cache.values()) if not sampler._transpiled_circ_cache else None
    p_b = cast(List[Dict[Parameter, float]], param_bindings)
    sampled_statefn_dicts = mod_sample_circuits(sampler,
                                                circuit_sfns=circs, 
                                                param_bindings=p_b, 
                                                para_op_pair=para_op_pair,
                                                )

    def replace_circuits_with_dicts(operator, param_index=0):
        if isinstance(operator, CircuitStateFn):
            assert param_index < len(sampled_statefn_dicts[id(operator)]), \
                f'id:{id(operator)}, ops:{len(sampled_statefn_dicts.keys())}, \
                    {len(sampled_statefn_dicts[id(operator)])}, {param_index}'
            return sampled_statefn_dicts[id(operator)][param_index]
        elif isinstance(operator, ListOp):
            return operator.traverse(
                partial(replace_circuits_with_dicts, param_index=param_index))
        else:
            return operator

    # store the operator we constructed, if it isn't stored already
    if op_id not in sampler._cached_ops.keys():
        op_cache = OperatorCache()
        op_cache.reduced_op_cache = sampler._reduced_op_cache
        op_cache.circuit_ops_cache = sampler._circuit_ops_cache
        op_cache.transpiled_circ_cache = sampler._transpiled_circ_cache
        op_cache.transpile_before_bind = sampler._transpile_before_bind
        op_cache.transpiled_circ_templates = sampler._transpiled_circ_templates
        sampler._cached_ops[op_id] = op_cache

    circ_len = [len(v) for v in sampled_statefn_dicts.values()]
    print("Circ len for each observable : {}".format(circ_len))
    op_to_paraidx = [0 for _ in range(len(operator))]

    result_op = []
    if return_as_list:
        if para_op_pair:
            for i in range(num_parameterizations):
                if i in para_op_pair.keys():
                    temp_oplist = []
                    for op_idx in para_op_pair[i]:
                        para_idx = op_to_paraidx[op_idx]
                        op = ComposedOp(replace_circuits_with_dicts(sampler._reduced_op_cache[op_idx], 
                                                                    param_index=para_idx))
                        temp_oplist.append(op)
                        op_to_paraidx[op_idx] += 1
                    result_op.append(SummedOp(temp_oplist))

                else:
                    temp_oplist = []
                    for op_idx in range(len(operator)):
                        para_idx = op_to_paraidx[op_idx]
                        op = ComposedOp(replace_circuits_with_dicts(sampler._reduced_op_cache[op_idx], 
                                                                    param_index=para_idx))
                        temp_oplist.append(op)
                        op_to_paraidx[op_idx] += 1

                    result_op.append(SummedOp(temp_oplist))
            assert op_to_paraidx == circ_len
        else:
            for i in range(num_parameterizations):
                result_op.append(replace_circuits_with_dicts(sampler._reduced_op_cache, param_index=i))
        return ListOp(result_op)

    else:
        return replace_circuits_with_dicts(sampler._reduced_op_cache, param_index=0)


def calculate_trasient_noise(exp_mode, Em_x_list, Er_x_list, E_new, ratio=1, method='amp_p', p_th=0):
    Er_x_list = np.array(Er_x_list)
    Em_x_list = np.array(Em_x_list)
    amp_th = 25 - len(Em_x_list)*5
    g_th = 0.3
    
    if exp_mode == "DISQ" and  method == "grad":
        Em_x_list_hsum = np.sum(Em_x_list)/len(Em_x_list)
        Er_x_list_hsum = np.sum(Er_x_list)/len(Er_x_list)
        E_new_sum = np.sum(E_new)
        G_new = E_new_sum - Er_x_list_hsum
        Gdf_new = E_new_sum -  Em_x_list_hsum
        abs_v = abs(G_new) + abs(Gdf_new)
        grad = G_new*Gdf_new

        th_v = g_th*abs(Gdf_new)
        grad_result = "FAIL" if (grad < 0 and abs_v > th_v) else "PASS"
        print('{}\nGrad estimation {:<5.3f}<0 and {:<5.3f}>{:<5.3f}: <{}>, G_new: {:<5.3f}, Gdf_new: {:<5.3f}'.format(
            "-------------------------------------------------------------------------",
            grad, abs_v, th_v, grad_result, G_new, Gdf_new))
        print('Er h-sum: {:<5.3f} | Em h-sum: {:<5.3f} | E_new_sum {:<5.3f}'.format(Er_x_list_hsum, Em_x_list_hsum, E_new_sum))
        print("-------------------------------------------------------------------------")
        if grad_result == "FAIL":
            return False, None
        elif grad_result == "PASS":
            return True, None
        else:
            raise Exception(f"Grad_result is <{grad_result}>, no match options")

    elif exp_mode == "DISQ" and "amp" in method:     # amplitude method (exp)
        Tr_n_list = Er_x_list - Em_x_list
        sign = np.sum(Er_x_list-Em_x_list)/abs(np.sum(Er_x_list-Em_x_list))
        Tr_n_list_hsum = np.sum(abs(Tr_n_list), axis=1)
        Em_x_list_hsum = np.sum(abs(Em_x_list), axis=1)
        Er_x_list_hsum = np.sum(abs(Er_x_list), axis=1)
        p_list = Tr_n_list_hsum/Em_x_list_hsum  
        p = np.mean(p_list)*100

        Ep_new = np.sum(E_new) - np.mean(Tr_n_list_hsum)*sign
        Ep_new_percent = np.sum(E_new) * (1 - np.mean(p_list)*sign)
        print(f'sign : {sign}, perecent : {Ep_new}, amp : {Ep_new_percent}')
        if method == "amp_p":
            Ep_new = Ep_new_percent
        amp_result = "FAIL" if p > amp_th else "PASS"
        print('{}\nAmplitude estimation: <{}>, noise_percent: {:<5.3f} %, th: {} %'.format(
            "-------------------------------------------------------------------------",
            amp_result, p, amp_th))
        print('Er h-sum: {} | Em h-sum: {} | Tr h-sum {}'.format(Er_x_list_hsum, Em_x_list_hsum, Tr_n_list_hsum))
        print("-------------------------------------------------------------------------")
        if amp_result == "FAIL":
            return False, Ep_new
        elif amp_result == "PASS":
            return True, Ep_new
        else:
            raise Exception(f"Amp_result is <{amp_result}>, no match options")
    else:
        raise Exception(f"This optimizer {exp_mode}, no mitgation options")
        


class SPSA_exp(Optimizer):
    def __init__(
        self,
        maxiter: int = 100,
        cal_c: float = 0.2,
        cal_a: float = 0.602,
        factor: float = 0, 
        exp_mode: str = 'DISQ',
        method: str = "amp", 
        blocking: bool = False,
        allowed_increase: Optional[float] = None,
        trust_region: bool = False,
        learning_rate: Optional[Union[float, np.array, Callable[[], Iterator]]] = None,
        perturbation: Optional[Union[float, np.array, Callable[[], Iterator]]] = None,
        last_avg: int = 1,
        resamplings: Union[int, Dict[int, int]] = 1,
        perturbation_dims: Optional[int] = None,
        second_order: bool = False,
        regularization: Optional[float] = None,
        hessian_delay: int = 0,
        lse_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        initial_hessian: Optional[np.ndarray] = None,
        callback: Optional[CALLBACK] = None,
        termination_checker: Optional[TERMINATIONCHECKER] = None,
    ) -> None:
        super().__init__()
        # general optimizer arguments
        self.maxiter = maxiter
        self.trust_region = trust_region
        self.callback = callback
        self.termination_checker = termination_checker

        # if learning rate and perturbation are arrays, check they are sufficiently long
        for attr, name in zip([learning_rate, perturbation], ["learning_rate", "perturbation"]):
            if isinstance(attr, (list, np.ndarray)):
                if len(attr) < maxiter:
                    raise ValueError(f"Length of {name} is smaller than maxiter ({maxiter}).")

        self.learning_rate = learning_rate
        self.perturbation = perturbation

        # fixed parameters
        self.cali_steps = 25
        self.exp_num = 1
        self.cal_c = cal_c
        self.cal_a = cal_a
        self.factor = factor
        self.method = method
        self.major_op_idx = None

        self.p_pos = 0
        self.m_pos = 1
        self.exp_pos = 0

        self.exp_mode = exp_mode 
        assert self.exp_mode == "DISQ" or self.exp_mode == "SPSA"

        # evaluation parameters
        self.nfun_ev = 0
        self.X, self.Y = [], []
        self.Y_mean = []
        self.maxargs = []
        self.predict_Y = []
        self.f_results = []
        
        # noise parameters
        self.noise = False
        self.noise_period = 0
        self.miti_his = []

        # exp rerun parameters
        self.valid = True
        self.rep_max = 10
        self.rep_cnt = 0
        self.r_idx = REF_LIST

        # SPSA specific arguments
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.last_avg = last_avg
        self.resamplings = resamplings
        self.perturbation_dims = perturbation_dims

        # 2-SPSA specific arguments
        if regularization is None:
            regularization = 0.01
        self.second_order = second_order
        self.hessian_delay = hessian_delay
        self.lse_solver = lse_solver
        self.regularization = regularization
        self.initial_hessian = initial_hessian

        # runtime arguments
        self._nfev = None  # the number of function evaluations
        self._smoothed_hessian = None  # smoothed average of the Hessians


    def calibrate(
        self,
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        c: float = 0.2, # default 0.2
        stability_constant: float = 0,
        target_magnitude: Optional[float] = None,  # 2 pi / 10
        alpha: float = 0.602,   # default 0.602
        gamma: float = 0.101,
        modelspace: bool = False,
        max_evals_grouped: int = 1,
    ) -> Tuple[Iterator[float], Iterator[float]]:
        
        logger.info("SPSA: Starting calibration of learning rate and perturbation.")
        if target_magnitude is None:
            target_magnitude = 2 * np.pi / 10

        dim = len(initial_point)

        # compute the average magnitude of the first step
        steps = self.cali_steps
        points = []
        for _ in range(steps):
            # compute the random directon
            pert = bernoulli_perturbation(dim)
            points += [initial_point + c * pert, initial_point - c * pert]

        losses, maxargs = _batch_evaluate(loss, points, max_evals_grouped, calibrate=True) 

        losses = losses[0]
        maxargs = maxargs[0]
        self.nfun_ev += steps
        avg_magnitudes = 0

        for i in range(steps):
            delta = losses[i][0] - losses[i][1]
            avg_magnitudes += np.abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes**2)
        else:
            a = target_magnitude / avg_magnitudes

        # compute the rescaling factor for correct first learning rate
        if a < 1e-10:
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude

        ''' Add calibrate in to database '''
        points_pairs = []
        for i in range(int(len(points)/2)):
            points_pairs.append([points[2*i], points[2*i+1]])

        self.X.extend(points_pairs)
        self.Y.extend(losses.tolist())
        self.Y_mean.extend(np.mean(losses, axis=1).tolist())
        self.predict_Y.extend(losses.tolist())
        self.maxargs.extend(maxargs.tolist())
        
        ''' set up the learning rate and perturbation '''
        logger.info("Finished calibration:")
        print(" -- Learning rate: a / ((A + n) ^ alpha) with a = {:.8f}, A = {}, alpha = {}"
            .format(a, stability_constant, alpha))
        print(" -- Perturbation : c / (n ^ gamma) with c = {}, gamma = {}".format(c, gamma))

        # set up the powerseries
        def learning_rate():
            return powerseries(a, alpha, stability_constant)
        def perturbation():
            return powerseries(c, gamma)
        def x_learning_rate(n):
            return a / ((n + stability_constant) ** alpha)
        def x_perturbation(n):
            return c / ((n + 0) ** gamma)

        return (learning_rate, perturbation, x_learning_rate, x_perturbation)


    @staticmethod
    def estimate_stddev(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        avg: int = 25,
        max_evals_grouped: int = 1,
    ) -> float:
        """Estimate the standard deviation of the loss function."""
        losses = _batch_evaluate(loss, avg * [initial_point], max_evals_grouped)
        return np.std(losses)

    @property
    def settings(self) -> Dict[str, Any]:
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        if callable(self.perturbation):
            iterator = self.perturbation()
            perturbation = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            perturbation = self.perturbation

        return {
            "maxiter": self.maxiter,
            "learning_rate": learning_rate,
            "perturbation": perturbation,
            "trust_region": self.trust_region,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
            "resamplings": self.resamplings,
            "perturbation_dims": self.perturbation_dims,
            "second_order": self.second_order,
            "hessian_delay": self.hessian_delay,
            "regularization": self.regularization,
            "lse_solver": self.lse_solver,
            "initial_hessian": self.initial_hessian,
            "callback": self.callback,
            "termination_checker": self.termination_checker,
        }


    def _point_sample(self, loss, x, eps, delta1, delta2):
        """A single sample of the gradient at position ``x`` in direction ``delta``."""
        points = [[x + eps * delta1, x - eps * delta1]]

        if self.second_order:
            points += [x + eps * (delta1 + delta2), x + eps * (-delta1 + delta2)]
            self._nfev += 2

        """ prepare rerun samples """      
        if self.exp_mode == "DISQ":
            ratio = [1/3, 1/3, 1/3] # ratio = 1, adjustable with advanced techniques
            assert np.sum(np.array(ratio)) == 1

            temp_r_idxes = REF_LIST if self.valid else self.r_idx
            para_op_pair = {}
            location = len(points) * 2

            if self.major_op_idx is not None:
                para_op_pair[0] = self.major_op_idx
                para_op_pair[1] = self.major_op_idx
            
            for i in temp_r_idxes:
                points.append(self.X[i])
                para_op_pair[location] = list(self.maxargs[i][self.p_pos].keys())
                location += 1
                para_op_pair[location] = list(self.maxargs[i][self.m_pos].keys())
                location += 1

            picked_pv = [list(self.maxargs[i][self.p_pos].values()) for i in temp_r_idxes]
            picked_mv = [list(self.maxargs[i][self.m_pos].values()) for i in temp_r_idxes]

            indicator_input = [picked_pv, picked_mv, temp_r_idxes, ratio]

            (values, t_values, t_maxargs, self.noise_period, p_flag, m_flag, predict_v) = \
                _batch_evaluate(loss, points, self._max_evals_grouped, calibrate=False, 
                                para_op_pair=para_op_pair, indicator_input=indicator_input)
        elif self.exp_mode == "SPSA":
            (values, t_values, t_maxargs, self.noise_period) = \
                  _batch_evaluate(loss, points, self._max_evals_grouped, calibrate=False)
            
        self.noise = self.factor > 0
        self.nfun_ev += 1

        """ transient noise mitigation evaluation """
        if self.exp_mode == "DISQ" and "amp" in self.method :
            condition = p_flag or m_flag
        elif self.exp_mode == "DISQ" and self.method == "grad":
            condition = p_flag and m_flag

        if self.exp_mode == "DISQ" :
            print("{}-{} rerun idx: {} (+): {} (-): {}"
                .format(self.exp_mode, self.method, temp_r_idxes, p_flag, m_flag), end="\t")
            if condition:
                self.r_idx, self.rep_cnt = temp_r_idxes, 0
                if not self.valid:
                    print('@ PASS rerun')
                else:
                    print("")
            else:
                self.r_idx = temp_r_idxes
                if not self.valid:
                    self.rep_cnt += 1
                    print(f"@ FAIL rerun - cnt: {self.rep_cnt}")
                else:
                    self.r_idx, self.rep_cnt = temp_r_idxes, 0
                    print(f"@ FAIL rerun - Reset cnt: {self.rep_cnt}")        
            self.valid = condition

        if self.exp_mode == "DISQ" :
            plus = t_values[self.exp_pos][self.p_pos]
            minus = t_values[self.exp_pos][self.m_pos]
        else:
            plus = values[self.exp_pos][self.p_pos]
            minus = values[self.exp_pos][self.m_pos] 

        gradient_sample = (plus - minus) / (2 * eps) * delta1

        hessian_sample = None
        if self.second_order:
            diff = (values[2] - plus) - (values[3] - minus)
            diff /= 2 * eps**2
            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return (t_values[self.exp_pos], 
                points[self.exp_pos], 
                t_maxargs[self.exp_pos], 
                gradient_sample, 
                hessian_sample)


    def _point_estimate(self, loss, x, eps, num_samples):
        """The gradient estimate at point x."""
        # set up variables to store averages
        value_estimate = 0
        gradient_estimate = np.zeros(x.size)
        hessian_estimate = np.zeros((x.size, x.size))
        # iterate over the directions
        deltas1 = [bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)]
        if self.second_order:
            deltas2 = [bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)]
        else:
            deltas2 = None

        for i in range(num_samples):
            delta1 = deltas1[i]
            delta2 = deltas2[i] if self.second_order else None

            value_sample, points, maxargs, gradient_sample, hessian_sample = \
                self._point_sample(loss, x, eps, delta1, delta2)
            value_estimate += value_sample
            gradient_estimate += gradient_sample
            if self.second_order:
                hessian_estimate += hessian_sample

        return (
            value_estimate / num_samples,
            points, 
            maxargs,
            gradient_estimate / num_samples,
            hessian_estimate / num_samples,
        )


    def _compute_update(self, loss, x, k, eps, lse_solver):
        # compute the perturbations
        if isinstance(self.resamplings, dict):
            num_samples = self.resamplings.get(k, 1)
        else:
            num_samples = self.resamplings

        # accumulate the number of samples
        value, points, maxargs, gradient, hessian = self._point_estimate(loss, x, eps, num_samples)

        # precondition gradient with inverse Hessian, if specified
        if self.second_order:
            smoothed = k / (k + 1) * self._smoothed_hessian + 1 / (k + 1) * hessian
            self._smoothed_hessian = smoothed
            if k > self.hessian_delay:
                spd_hessian = _make_spd(smoothed, self.regularization)
                # solve for the gradient update
                gradient = np.real(lse_solver(spd_hessian, gradient))

        return value, points, maxargs, gradient


    def update_database(self, fx_estimate, org_points, maxargs):
        self.X.append(org_points)
        self.Y.append(fx_estimate.tolist())
        self.maxargs.append(maxargs.tolist())
        self.Y_mean.append(np.mean(fx_estimate))


    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required

        if self.learning_rate is None and self.perturbation is None:
            print("\n{0}\n{1}\n{0}\n".
               format("===============================================================", 
               f"{self.exp_mode} - Do calibration : c = {self.cal_c}, alpha = {self.cal_a}"))

            get_eta, get_eps, xeta, xeps = self.calibrate(fun, 
                                                          x0, 
                                                          c=self.cal_c, 
                                                          alpha=self.cal_a, 
                                                          max_evals_grouped=self._max_evals_grouped)
        else:
            get_eta, get_eps = _validate_pert_and_learningrate(self.perturbation, self.learning_rate)
        eta, eps = get_eta(), get_eps()

        if self.lse_solver is None:
            lse_solver = np.linalg.solve
        else:
            lse_solver = self.lse_solver

        # prepare some initials
        x = np.asarray(x0)
        if self.initial_hessian is None:
            self._smoothed_hessian = np.identity(x.size)
        else:
            self._smoothed_hessian = self.initial_hessian

        self._nfev = 0
 
        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = fun(x)

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(
                    fun, x, max_evals_grouped=self._max_evals_grouped
                )

        print("\n{0}\n{1}\n{0}\n".
               format("===============================================================", 
               f"{self.exp_mode} - Starting optimization"))
        start = time()

        # keep track of the last few steps to return their average
        last_steps = deque([x])

        # use a local variable and while loop to keep track of the number of iterations
        # if the termination checker terminates early
        cnt = 0
        while cnt < self.maxiter:
            iteration_start = time()
            # compute update
            next_eps, next_eta = next(eps), next(eta)
            fx_estimate, org_points, maxargs, update = \
                self._compute_update(fun, x, cnt, next_eps, lse_solver)
            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:  # stop from dividing by 0
                    update = update / norm

            update = update * next_eta
            x_next = x - update
            fx_next = None

            # blocking
            if self.blocking:
                self._nfev += 1
                fx_next = fun(x_next)

                if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                    if self.callback is not None:
                        self.callback(
                            self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(update),  # size of the update step
                            False,
                        )  # not accepted

                    logger.info(
                        "Iteration %s/%s rejected in %s.",cnt, self.maxiter + 1, time()-iteration_start,
                    )
                    continue
                fx = fx_next

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = fun(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(update),  # size of the update step
                    True,
                )  # accepted

            ''' DISQ iteration validation (PASS: save, FAIL: rerun) '''
            if self.exp_mode == "DISQ" :
                if self.valid:
                    cnt += 1
                    self.update_database(fx_estimate, org_points, maxargs)
                else:
                    self.miti_his.append(self.nfun_ev)
                    if self.rep_cnt < self.rep_max:
                        x_next = x      # rerun x
                    else:
                        cnt += 1
                        self.update_database(fx_estimate, org_points, maxargs)
                        self.valid = True   # reset valid flag to stop rerun
                        print("{0}\n{1}\n{0}".format('********************************************',
                            'Rerun reaches to the repeat maximum'))
            else:
                cnt += 1
                self.update_database(fx_estimate, org_points, maxargs)

            # update parameters
            x = x_next
            print(f"Iteration {cnt}/{self.maxiter} done in {time()-iteration_start} (s)\n\n")

            # update the list of the last ``last_avg`` parameters
            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

            if self.termination_checker is not None:
                fx_check = fx_estimate if fx_next is None else fx_next
                if self.termination_checker(
                    self._nfev, x_next, fx_check, np.linalg.norm(update), True
                ):
                    logger.info("terminated optimization at {k}/{self.maxiter} iterations")
                    break

        logger.info("SPSA %s: Finished in %s (s)", self.exp_mode, time() - start)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        result = OptimizerResult()
        result.x = x
        result.fun = fun(x, output_mode=True)[0]    # ouptut in np.array type
        result.nfev = self._nfev
        result.nit = cnt
        self.nfun_ev += 1
        self.f_results = result
        return result


    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }





class custom_VQE(VariationalAlgorithm, MinimumEigensolver):
    """ The Variational Quantum Eigensolver algorithm. """
    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        maxiter: int = 1, 
        optimizer: Optional[Union[Optimizer, Minimizer]] = None,
        initial_point: Optional[np.ndarray] = None,
        gradient: Optional[Union[GradientBase, Callable]] = None,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
        quantum_instance: Optional[Union[QuantumInstance, Backend]] = None,
        threshold: Optional[Union[int, float]] = 0.9,
        method: str = "amp",
        prime_th: float = 0.7,
    ) -> None:
        """
        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.`
            quantum_instance: Quantum Instance or Backend
        """
        super().__init__()
        self.maxiter = maxiter
        self.nfun_ev = 0
        self.optimizer = optimizer
        
        self.factor_list = [0.95, 0.9, 0.875, 0.85, 0.8]
        self.prime_th = prime_th
        self.method = method

        ''' added for transient noise mitigation '''
        self.n_period = 0
        self.th = threshold
        self.noise_his = []
        self.minor_obs_his = []
        self.exe_time_his = []
        self.skip_time_his = []
        self.act_time_his = []
        self.internal_time_his = []
        self.act_internal_time_his = []
        self.op_to_paraidx_list = []
        self.major_op_idx = None

        self.factor = self.optimizer.factor
        self.cali_steps = self.optimizer.cali_steps
        self.exp_num = self.optimizer.exp_num
        self.exp_pos = self.optimizer.exp_pos
        self.p_pos = self.optimizer.p_pos
        self.m_pos = self.optimizer.m_pos
        self.exp_mode = self.optimizer.exp_mode
        self.iter = self.optimizer.maxiter
        self.xr_pos = self.exp_num

        ''' standard setups '''
        self.ansatz = ansatz
        self.callback = callback
        self.gradient = gradient
        self.quantum_instance = quantum_instance
        self.expect_op = None

        if quantum_instance is not None:
            self.circuit_sampler = CircuitSampler(quantum_instance)
        else:
            self.circuit_sampler = None

        self._initial_point = None
        self.initial_point = initial_point
        logger.info(self.print_settings())

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray):
        """Sets initial point"""
        self._initial_point = initial_point

    def _check_operator_ansatz(self, operator: OperatorBase):
        """Check that the number of qubits of operator and ansatz match."""
        if operator is not None and self.ansatz is not None:
            if operator.num_qubits != self.ansatz.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self.ansatz.num_qubits = operator.num_qubits
                except AttributeError as ex:
                    raise AlgorithmError(
                        "The number of qubits of the ansatz does not match the "
                        "operator, and the ansatz does not allow setting the "
                        "number of qubits using `num_qubits`."
                    ) from ex

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = f"Algorithm: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                elif key[1:] == 'ansatz':
                    params += f"-- {key[1:]}:\n{value}\n"
                else:
                    params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    def print_settings(self):
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.__class__.__name__
        )
        ret += f"{self.setting}"
        ret += "===============================================================\n"
        if self.ansatz is not None:
            ret += "{}".format(self.ansatz.draw(output="text"))
        else:
            ret += "ansatz has not been set"
        ret += "\n===============================================================\n"
        if callable(self.optimizer):
            ret += "Optimizer is custom callable\n"
        else:
            ret += f"{self.optimizer.setting}"
        ret += "===============================================================\n"
        return ret

    def construct_expectation(
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Union[OperatorBase, Tuple[OperatorBase, ExpectationBase]]:
        if operator is None:
            raise AlgorithmError("The operator was never provided.")

        self._check_operator_ansatz(operator)
        wave_function = self.ansatz.assign_parameters(parameter)
        ansatz_circuit_op = CircuitStateFn(wave_function)

        expectation = ExpectationFactory.build(
            operator=operator,
            backend=self.quantum_instance,
            include_custom=False,
        )
        observable_meas = expectation.convert(StateFn(operator, is_measurement=True))
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()
        print(f'expect_op')
        return (expect_op, expectation) if return_expectation else expect_op

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def create_VQEResult(self, opt_result):
        result = VQEResult()
        result.optimal_point = opt_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        result.optimal_value = opt_result.fun
        result.cost_function_evals = opt_result.nfev
        result.eigenvalue = opt_result.fun + 0j
        return result


    def compute_minimum_eigenvalue(
        self, 
        operator: OperatorBase, 
        aux_operators: Optional[ListOrDict[OperatorBase]] = None
    ) -> VQEResult:

        super().compute_minimum_eigenvalue(operator, aux_operators)
        if self.quantum_instance is None:
            raise AlgorithmError("Require Backend to run the VQE.")
        self.quantum_instance.circuit_summary = True

        # this sets the size of the ansatz, so it must be called before the initial point validation
        self._check_operator_ansatz(operator)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)
        
        ev_fuc = self.get_energy_evaluation(operator)
        opt_result = self.optimizer.minimize(ev_fuc, initial_point)
        out_results = [self.create_VQEResult(opt_result)]
        return out_results


    def get_energy_evaluation(
        self,
        operator: OperatorBase,
        return_expectation: bool = False,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError("The ansatz must be parameterized, but has 0 free parameters.")

        ansatz_params = self.ansatz.parameters
        expect_op, expectation = self.construct_expectation(ansatz_params, operator, return_expectation=True)
        expect_op_num = len(expect_op)
        self.expect_op = expect_op
        

        print('{0}\nExpect_op info\n# observable: {1}'.format("-----------------------------", expect_op_num))
        coef_opidx_dict = {}
        for i, op in enumerate(expect_op):
            coef_list = op.oplist[0].primitive.coeffs
            coef_opidx_dict[i] = sum(abs(coef_list))
        
        sorted_coef_opidx = sorted(coef_opidx_dict.items(), key=lambda x:x[1], reverse=True)
        coef_sum = sum(dict(sorted_coef_opidx).values())
        print(f'Total coeff sum : {coef_sum}\n{sorted_coef_opidx}')
        prime_list = []
        op_percent = 0
        if self.prime_th >= 1:
            prime_list = list(range(expect_op_num))
            self.major_op_idx = prime_list
            self.optimizer.major_op_idx = self.major_op_idx
        else:
            for value in sorted_coef_opidx:
                op_percent += value[1]/coef_sum
                prime_list.append(value[0])
                if op_percent >= self.prime_th:
                    print('op list: {0}/{1}, percent: {2} > {3}'
                        .format(len(prime_list), expect_op_num, op_percent, self.prime_th))
                    print(f'Primes : {prime_list}')
                    prime_list = sorted(prime_list)
                    self.major_op_idx = prime_list
                    self.optimizer.major_op_idx = self.major_op_idx
                    break
        assert isinstance(self.optimizer, SPSA_exp)


        def energy_eva_fuc(parameters, calibrate=False, para_op_pair=None, indicator_input=None, output_mode=False):
            if calibrate:
                self.nfun_ev += self.cali_steps
            else:
                self.nfun_ev += 1
                if self.th <= 0.95:
                    self.th +=  0.1 / self.iter

            print('{0}  func env < {1} >  {0}'.format("@@@@@@@@@@@@@@@", self.nfun_ev))
            parameter_sets = np.reshape(parameters, (-1, num_parameters))
            param_bindings = dict(zip(ansatz_params, parameter_sets.transpose().tolist()))

            if output_mode:
                sampled_expect_op = self.circuit_sampler.convert(expect_op, params=param_bindings)
                energys = np.real(sampled_expect_op.eval())
                return energys

            if calibrate:
                sampled_expect_op = modconvert(self.circuit_sampler, 
                                                expect_op, 
                                                param_bindings)
                energys, energys_prime = mod_eval_with_ob(sampled_expect_op, self.major_op_idx)
                """ Identify the method in execution """
                print('{0}\nPrime ob: {1}, with prime th {2} %\n{0}'
                    .format("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", self.major_op_idx, self.prime_th))                    
                    
                energys = energys.reshape((self.exp_num, self.cali_steps, 2))
                energys_prime = energys_prime.reshape((self.exp_num, self.cali_steps, 2))

                if self.callback is not None: 
                    self.callback(energys, calibrate)
                return energys, energys_prime


            if para_op_pair:
                params_exp = np.reshape(parameters[self.exp_pos], (-1, num_parameters))
                param_bindings_exp = dict(zip(ansatz_params, params_exp.transpose().tolist()))
                assert len(parameter_sets) == len(para_op_pair), f'para: {len(parameter_sets)}, obs: {len(para_op_pair)}'

            """ EXECUTION STEP 1 """
            execution_1_start = time()
            sampled_expect_op = modconvert(self.circuit_sampler, 
                                            expect_op, 
                                            param_bindings, 
                                            para_op_pair, 
                                            )
            exe_time = time() - execution_1_start
            print(f'step1 exe time : {exe_time}')
            if self.exp_mode == "DISQ":
                energys, energys_prime = mod_eval_exp_step(sampled_expect_op, self.major_op_idx)
            else:
                energys, energys_prime = mod_eval_exp_step(sampled_expect_op, self.major_op_idx)
                # mod_eval(sampled_expect_op)

            bsize = len(energys)     # [e, qi_r+] each 2 samples
            energys  = energys.reshape((int(bsize/2), 2))
            energys_prime = energys_prime.reshape((int(bsize/2), 2))
            t_energys, t_energys_prime = self.noise_model(energys=energys, primes=energys_prime)
            
            if self.exp_mode == "DISQ":
                picked_pv = indicator_input[0]
                picked_mv = indicator_input[1]
                r_idxes   = indicator_input[2]
                ratio     = indicator_input[3]

                rerun_pv = [list(t_energys_prime[i][self.p_pos].values()) for i in range(self.xr_pos, self.xr_pos+len(r_idxes))]
                new_pv = list(t_energys_prime[self.exp_pos][self.p_pos].values())
                p_flag, Ep_y_p = calculate_trasient_noise(self.exp_mode, picked_pv, rerun_pv, new_pv, ratio, self.method, self.prime_th*10)
                
                rerun_mv = [list(t_energys_prime[i][self.m_pos].values()) for i in range(self.xr_pos, self.xr_pos+len(r_idxes))]
                new_mv = list(t_energys_prime[self.exp_pos][self.m_pos].values())
                m_flag, Ep_y_m = calculate_trasient_noise(self.exp_mode, picked_mv, rerun_mv, new_mv, ratio, self.method, self.prime_th*10)

                if "amp" in self.method :
                    valid = (p_flag or m_flag)
                elif self.method == "grad":
                    valid = (p_flag and m_flag)

                if valid:
                    """ EXECUTION STEP 2 """
                    para_op_pair_minor = {}
                    for i in range(len(params_exp)):
                        para_op_pair_minor[i]= np.setdiff1d(list(range(expect_op_num)), para_op_pair[i]).tolist()
                    # print(para_op_pair_minor)

                    execution_2_start = time()
                    sampled_expect_op_minor = modconvert(self.circuit_sampler, 
                                                         expect_op, 
                                                         param_bindings_exp, 
                                                         para_op_pair_minor, 
                                                        )
                    exe_time_2 = time() - execution_2_start
                    exe_time += exe_time_2
                    print(f'step2 exe time : {exe_time_2}')                                     
                    minor_energys, _ = mod_eval(sampled_expect_op_minor)
                    print(f'minor_energys: {minor_energys}')
                    t_minor_energys, _ = self.noise_model(minor_energys, np_dec=False)
                    t_energys[self.exp_pos] += t_minor_energys
                    self.energy_print(t_energys[0:self.xr_pos], 'After add energy:')
                    self.act_time_his.append(exe_time)
                else:
                    self.skip_time_his.append(exe_time)

                print(f'total exe time : {exe_time}')    
                self.exe_time_his.append(exe_time)
                if self.callback is not None: 
                    self.callback(energys, calibrate)

                return (energys, 
                        t_energys, 
                        t_energys_prime, 
                        self.n_period, 
                        p_flag, m_flag,
                        [Ep_y_p, Ep_y_m],
                        )
            elif self.exp_mode == "SPSA":
                self.exe_time_his.append(exe_time)
                if self.callback is not None: 
                    self.callback(energys, calibrate)
                return (energys, t_energys, t_energys_prime, self.n_period)

        return (energy_eva_fuc, expectation) if return_expectation else energy_eva_fuc 
    

    def energy_print(self, value, output_str, prime_flag=False, ending="\n"):
        assert(type(output_str) == str)
        if prime_flag:
            iter = np.nditer(value, flags=['c_index', "refs_ok"])
            for e in iter:
                for i, (k, v) in enumerate(e.item().items()):   # e -> numpy.ndarray, e.item() -> dict
                    if i+1 == len(e.item().items()):
                        if iter.index+1 == iter.itersize:
                            temp_str = f'{v:.4f}({k})'
                        elif iter.index%2 == 0:
                            temp_str = f'{v:.4f}({k}) | '
                        else:
                            temp_str = f'{v:.4f}({k})\n'
                    else:
                        temp_str = f'{v:.4f}({k}), '
                    output_str += temp_str
        else:
            iter = np.nditer(value, flags=['c_index'])
            for e in iter:
                if iter.index+1 == iter.itersize:
                    temp_str = f'{e:.6f}'
                elif iter.index%2 == 0:
                    temp_str = f'{e:.6f}, '
                else:
                    temp_str = f'{e:.6f} | '
                output_str += temp_str
        print(output_str, end=ending)


    def noise_model(self, energys, primes=None, np_dec=True):
        t_energys  = energys
        t_primes = primes

        """ Insert transient noise """
        if NOISE_ENABLE:
            noise_map = {
                19: [7, 0.275],
                108: [20, 0.2],
            }
            n_iter = self.nfun_ev - self.cali_steps
            if n_iter in noise_map.keys() and self.n_period == 0:
                if isinstance(noise_map[n_iter], list):
                    self.n_period = noise_map[n_iter][0]
                    f = noise_map[n_iter][1]
                    self.optimizer.factor = f/self.n_period
                    self.factor = 1-f

            if self.n_period == 0:
                self.optimizer.factor = 0
            
        if self.n_period > 0:
            print("{0} < WARNING TRANSIENT NOISE > {0}".format("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"))            
            t_energys = energys * self.factor
            self.energy_print(energys[0:self.xr_pos], 'energys  : ', ending=" | ")
            self.energy_print(t_energys[0:self.xr_pos], 'T energys: ')

            if primes is not None and len(t_primes) > 1:
                t_primes = copy.deepcopy(primes)
                for e in np.nditer(t_primes, flags=["refs_ok"], op_flags=['writeonly']):
                    e = e.item()
                    e.update((k, v * self.factor) for k, v in e.items())
                self.energy_print(primes[self.xr_pos:], 'primes  : ', prime_flag=True)
                self.energy_print(t_primes[self.xr_pos:], 'T primes: ', prime_flag=True)

            print(f'Noise period: {self.n_period}, mag: {self.factor}, th: {self.th}, cnts: {len(self.noise_his)}')

            if np_dec:
                if self.factor != 1:
                    self.noise_his.append(self.nfun_ev)
                self.n_period -= 1
                assert self.n_period >=0, f'noise period is negative: {self.n_period}'
        else:
            print('{0} < No transient noise effects > {0}'.format("~~~~~~~~~~~~~~~~~~~~~~~"))
            self.energy_print(t_energys[0:self.xr_pos], 'energys: ')
            if t_primes is not None and len(t_primes) > 1:
                self.energy_print(t_primes[self.xr_pos:], 'primes : ', prime_flag=True)
        return t_energys, t_primes




def main(backend, user_messenger, **kwargs):
    """Entry function."""
    mandatory = {"ansatz", "operator", "output_dir"}
    missing = mandatory - set(kwargs.keys())
    if len(missing) > 0:
        raise ValueError(f"The following mandatory arguments are missing: {missing}.")

    ansatz = kwargs["ansatz"]
    operator = kwargs["operator"]
    output_dir = kwargs["output_dir"]
    seed = kwargs.get("seed", None)
    shots = kwargs.get("shots", 1024)
    iteration = kwargs.get("iteration", 20)
    optimizer_name = kwargs.get("optimizer", 'DISQ')
    initial_point = kwargs.get("initial_point", None)
    threshold = kwargs.get("threshold", 0.9)
    method = kwargs.get("method", "amp_a")
    prime_th = kwargs.get("prime_th", 0.9)

    cal_c = kwargs.get("c", 0.2)
    cal_a = kwargs.get("alpha", 0.602)
    factor = kwargs.get("factor", 0)
    optimizer = SPSA_exp(maxiter=iteration, 
                         cal_c=cal_c, 
                         cal_a=cal_a, 
                         factor=factor,
                         exp_mode=optimizer_name,
                         method=method,
                         )    

    np.random.seed(seed)
    algorithm_globals.random_seed = seed

    print(f'is simulator: {backend.configuration().simulator}\nInput Kwargs:')
    for k, v in kwargs.items():
        if k == 'ansatz' or k == 'output_dir' or k == 'factor' or k == 'noise_model':
            continue
        print(f'{k}: {v}')
    
    print("\n{0}\n{1}{2}\n{3}{4}\n{5}{6}\n{7}{8}\n{0}\n".
           format("===============================================================", 
           "BACKEND: "     , backend.name, 
           "OUTPUT DIR: "  , output_dir, 
           "NOISE FACTOR: ", factor,
           "OPTIMIZER: "   , optimizer_name))
    
    if backend.configuration().simulator is True:
        n_model = kwargs.get("noise_model", None)
        if n_model:
            print("\n{0}\n{1}{2}\n{0}\n".
            format("===============================================================", 
            "NOISE MODEL: ", n_model))
            coupling_map = backend.configuration().coupling_map
            basis_gates = n_model.basis_gates
            _quantum_instance = QuantumInstance(backend, 
                                                shots=shots, 
                                                noise_model=n_model,
                                                coupling_map=coupling_map,
                                                basis_gates=basis_gates,
                                                seed_transpiler=seed, 
                                                seed_simulator=seed
                                                )
        else:
            print("\n{0}\n{1}\n{0}\n".
            format("===============================================================", 
            "NO NOISE MODEL"))
            _quantum_instance = QuantumInstance(backend, 
                                                shots=shots, 
                                                seed_transpiler=seed, 
                                                seed_simulator=seed
                                                )
    else:
        _quantum_instance = QuantumInstance(backend, shots=shots)

    # verify the initial point
    if initial_point == "random" or initial_point is None:
        initial_point = np.random.random(ansatz.num_parameters)
    elif len(initial_point) != ansatz.num_parameters:
        raise ValueError("Mismatching number of parameters and initial point dimension.")

    exp_num = 1
    all_value = [[] for _ in range(exp_num)]
    def call_back(energy, calibrate):
        if calibrate:            
            for i in range(vqe_model.optimizer.cali_steps):
                for a, e in zip(all_value, energy):
                    a.extend(e[i].tolist())
        else:
            for a, e in zip(all_value, energy):
                a.extend(e.tolist())

    # construct the VQE instance
    start_time = time()
    vqe_model = custom_VQE(ansatz=ansatz,
                           maxiter=iteration, 
                           optimizer=optimizer,
                           callback=call_back,
                           initial_point=initial_point, 
                           quantum_instance=_quantum_instance,
                           threshold=threshold,
                           method=method,
                           prime_th=prime_th,
                           )
    result = vqe_model.compute_minimum_eigenvalue(operator=operator)
    total_time = time()-start_time
    logger.warning(f'total running time: {total_time}')

    # compare mitigation and noise history
    noise_history = np.array(vqe_model.noise_his)
    mitigation_history = np.array(vqe_model.optimizer.miti_his)
    execution_time_history = np.array(vqe_model.exe_time_his)
    act_t_his = np.array(vqe_model.act_time_his)
    inter_t_his = np.array(vqe_model.internal_time_his)
    act_inter_t_his = np.array(vqe_model.act_internal_time_his)
    op_to_paraidx_list = np.array(vqe_model.op_to_paraidx_list)

    max_args = vqe_model.optimizer.maxargs
    sum_args_list = []
    abs_args_list = []
    sum_args = np.array(copy.deepcopy(max_args))
    for e in np.nditer(sum_args, flags=["refs_ok"], op_flags=['writeonly']):
        e = e.item()
        abs_args_list.append(sum(np.abs(np.array(list(e.values())))))
        sum_args_list.append(sum(e.values()))
    num = len(np.array(vqe_model.optimizer.Y))
    sum_args_list = np.array(sum_args_list).reshape((num, 2))
    abs_args_list = np.array(abs_args_list).reshape((num, 2))

    ''' Saving results '''
    other_data = {"noise_history" : noise_history,
                  "mitigation_history" : mitigation_history,
                  "prime_idx": vqe_model.major_op_idx,
                  "nfun_ev" : vqe_model.nfun_ev, 
                  "total_time" : total_time,
                  "expect_op_num": len(vqe_model.expect_op),

                  "execution_time_history" : execution_time_history,
                  "actual_execution_time_history": act_t_his,
                  "internal_time_his": inter_t_his,
                  "act_internal_time_his": act_inter_t_his,
                  "op_to_paraidx_list" : op_to_paraidx_list,
                }
    
    history = { "values"     : np.array(vqe_model.optimizer.Y).flatten(), 
                "values_a"   : np.array(vqe_model.optimizer.Y_mean), 
                "values_all" : all_value,
                "sum_args_list": sum_args_list,
                "abs_args_list": abs_args_list,
            }

    serialized_result = {
        "optimizer_history": history,
        "result_list": [result[i].eigenvalue.real for i in range(len(result))],
        "other_data": other_data,
    }
    return serialized_result