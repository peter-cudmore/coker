from typing import Tuple, Type, Union
from coker import Dimension, Function, VectorSpace

from coker.backends.backend import Backend, ArrayLike

from coker.backends.casadi.casadi import *
from coker.backends.casadi.optimiser import build_optimisation_problem

from coker.backends.casadi.variational_solver import create_variational_solver
from coker.dynamics import VariationalProblem


scalar_types = (float, int)


class CasadiBackend(Backend):
    def to_numpy_array(self, array: Union[ca.MX, ca.DM]) -> ArrayLike:
        if isinstance(array, ca.MX):
            try:
                return array.to_DM().toarray()
            except RuntimeError:
                pass
        elif isinstance(array, ca.DM):
            return array.toarray()
        try:
            return ca.evalf(array).toarray()
        except RuntimeError:
            pass

        raise ValueError(f"Cannot convert {array} to a numpy array")

    def to_backend_array(self, array):
        if array is None:
            return ca.DM()
        if isinstance(array, scalar_types):
            return ca.DM(array)
        if array.shape == (1, 1):
            return ca.DM(array[0, 0])
        elif array.shape == (1,):
            return ca.DM(array[0])
        elif len(array.shape) >= 2:
            result = ca.DM.zeros(*array.shape)
            with np.nditer(
                array, flags=["multi_index"], op_flags=["readonly"]
            ) as it:
                for v in it:
                    if v != 0:
                        key = tuple(it.multi_index)
                        result[key] = v

        elif len(array.shape) == 1 and array.shape[0] > 1:
            (n,) = array.shape
            result = ca.DM(n, 1)
            for i, v in enumerate(array):
                if v != 0:
                    result[i] = v

        else:
            raise NotImplementedError(
                f"Don't know how to convert {array} to a casadi array"
            )

        assert not isinstance(result, Tracer)

        return result

    def call(self, op, *args) -> ArrayLike:
        try:
            result: ca.DM = impls[op](*args)
            assert result.is_regular(), f"{op}({args}) =  {result}"

            return result
        except KeyError:
            pass

        if isinstance(op, tuple(parameterised_impls.keys())):
            result = call_parameterised_op(op, *args)
            assert result.is_regular(), f"{op}({args}) =  {result}"
            return result

        if isinstance(op, ReshapeOP):

            (arg,) = args
            shape = op.newshape
            if len(shape) == 1:
                shape = (1, *shape)

            return ca.reshape(arg, shape)

        raise NotImplementedError(f"{op} is not implemented")

    def native_types(self) -> Tuple[Type]:
        pass

    def reshape(self, array: ArrayLike, dim: Dimension) -> ArrayLike:
        if dim.is_scalar():
            return array

        if dim.is_vector():
            shape = (*dim, 1)
        else:
            shape = tuple(dim)

        if isinstance(array, (ca.MX, ca.DM)):
            return ca.reshape(array, *shape)
        if isinstance(array, np.ndarray):
            return ca.reshape(array, *shape)
        raise NotImplementedError

    def lower(self, function: Function):
        assert not any(
            isinstance(shape, FunctionSpace)
            for shape in function.input_shape()
        ), "Cannot lower a partially evaluated function."
        return lower(function.tape, function.output)

    def evaluate(self, function: Function, inputs: ArrayLike):
        workspace = {}

        for idx, (space, arg) in enumerate(
            zip(function.input_shape(), inputs)
        ):
            assert not isinstance(arg, Tracer)
            index = function.tape.input_indicies[idx]
            if isinstance(arg, np.ndarray):
                workspace[index] = self.to_backend_array(arg)

            else:
                workspace[index] = arg

        y = substitute(function.output, workspace)
        outs = []
        for y_i, output_tracer in zip(y, function.output):
            try:
                y_result = self.to_numpy_array(y_i)
                if output_tracer.dim.is_scalar():
                    if y_result.shape == (1, 1):
                        outs.append(float(y_result[0, 0]))
                    elif y_result.shape == (1,):
                        outs.append(float(y_result[0]))
                    else:
                        raise ValueError("Expected a scalar", y_result)
                else:
                    outs.append(y_result.reshape(output_tracer.shape))

            except ValueError:
                outs.append(y_i)

        return outs

    def to_array(self, arg: Union[ca.MX, ca.DM]):

        return self.to_numpy_array(arg)

    def build_optimisation_problem(
        self, cost, constraints, parameters, outputs, initial_conditions
    ):
        return build_optimisation_problem(
            cost, constraints, parameters, outputs, initial_conditions
        )

    def create_variational_solver(self, problem: VariationalProblem):
        return create_variational_solver(problem)

    def evaluate_integrals(
        self,
        functions,
        initial_conditions,
        end_point: float,
        inputs,
        solver_parameters=None,
    ):
        dxdt, g, dqdt = functions

        is_dae = dqdt is not Noop()
        has_quadrature = g is not Noop()
        x0, z0, q0 = (self.to_backend_array(a) for a in initial_conditions)
        if isinstance(end_point, (int, float)):
            if end_point == 0:
                return x0, z0, q0

            t_eval = [end_point]
        else:
            t_eval = end_point

        u, p = inputs
        p = self.to_backend_array(p)
        t = ca.MX.sym("t")
        x = ca.MX.sym("x", x0.shape)
        z = ca.MX.sym("z", z0.shape)

        dx_sym = dxdt(t, x, z, u, p)

        if is_dae:
            q = ca.MX.sym("q", q0.shape)
            q0 = ca.DM.zeros(q.shape)
            dq_sym = dqdt(t, x, z, u, p)
            xq = ca.vertcat(x, q)
            xq0 = ca.vertcat(x0, q0)
            xq_to_x_q = ca.Function("xq_to_x_q", [xq], [x, q])
            dxq = ca.vertcat(dx_sym, dq_sym)
        else:
            xq = x
            xq0 = x0
            xq_to_x_q = ca.Function("xq_to_x_q", [xq], [x])
            dxq = ca.vertcat(dx_sym)

        initial_conditions = {
            "x0": x0,
        }
        dae = {
            "x": xq,
            "ode": dxq,
        }
        if is_dae:
            dae["z"] = g(t, x, z, u, p)
            initial_conditions["z0"] = z0

        solver = ca.integrator("solver", "idas", dae, 0, t_eval, {})
        xq_final = solver(x0=xq0, z0=z0)

        if has_quadrature:
            x_final, q_final = xq_to_x_q(xq_final["xf"])
        else:
            x_final = xq_to_x_q(xq_final["xf"])
            q_final = None
        if is_dae:
            z_final = xq_final["zf"]
        else:
            z_final = None

        return x_final, z_final, q_final
