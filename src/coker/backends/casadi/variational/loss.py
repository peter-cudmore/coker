"""Loss lowering for CasADi variational transcriptions."""

import casadi as ca
from coker.algebra.graph import Tracer
from coker.backends.casadi.lower import substitute


from .factory import _TranscriptionFactory
from .symbolic_path import SymbolicPolyCollection


def _lower_loss(
    factory: "_TranscriptionFactory",
    poly_collection: "SymbolicPolyCollection",
    duration,
):
    """Lower the problem loss against one transcription's solution proxies."""
    problem = factory.problem

    def normalized_time(time):
        return time if factory.free_horizon else time / duration

    def state_proxy(time):
        return factory.proj_x @ poly_collection(normalized_time(time))

    def input_proxy(time):
        return factory.control_eval(normalized_time(time))

    def solution_proxy(*args):
        if len(args) == 1:
            time, control_val, p_val = (
                args[0],
                factory.control_eval,
                factory.p,
            )
        elif factory.control_factory is None:
            time, p_val = args
            control_val = factory.control_eval
        else:
            time, control_val, p_val = args
        tau = normalized_time(time)
        u_val = control_val(tau)
        inner = poly_collection(tau)
        x_tau = factory.proj_x @ inner
        z_tau = factory.proj_z @ inner
        q_tau = factory.proj_q @ inner
        (y_val,) = factory.casadi.evaluate(
            problem.system.y,
            [time, x_tau, z_tau, u_val, factory.proj_p @ p_val, q_tau],
        )
        return y_val

    if isinstance(problem.loss, Tracer):
        values = {
            "t": duration,
            "t_final": duration,
            "t_0": ca.DM.zeros(1, 1),
            "_state": state_proxy,
            "p": factory.p,
            "_output": solution_proxy,
        }
        if factory.control_factory is not None:
            values[problem.system.inputs.name] = input_proxy
        workspace = {
            index: values[name]
            for index, name in zip(
                problem.loss.tape.input_indicies,
                problem.loss.tape.input_names,
            )
            if name in values
        }
        (cost,) = substitute([problem.loss], workspace)
    elif factory.control_factory is None:
        (cost,) = factory.casadi.evaluate(
            problem.loss, [solution_proxy, factory.p]
        )
    else:
        (cost,) = factory.casadi.evaluate(
            problem.loss,
            [solution_proxy, factory.control_factory, factory.p],
        )
    return cost
