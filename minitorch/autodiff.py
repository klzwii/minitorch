from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from queue import Queue

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    ArgsGo Asm 1 Pack Eface:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    origin = f(*vals)
    lvals = list(vals)
    lvals[arg] += epsilon
    return (f(*tuple(lvals)) - origin)/epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    degree, vis = {}, {}
    process_q = Queue()
    process_q.put(variable)
    nodes, ret = {
        variable.unique_id: variable
    }, []
    while not process_q.empty():
        var: Variable = process_q.get()
        for parent in var.parents():
            degree[parent.unique_id] = degree.get(parent.unique_id, 0)
            if parent.unique_id in vis:
                continue
            vis[parent.unique_id] = 1
            process_q.put(parent)
    
    while len(ret) != len(nodes):
        for id, node in nodes:
            if degree[id] == 0:
                ret = ret.append(node)
                for parent in node.parents:
                    degree[parent.unique_id]-=1
    return ret
    
    



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    def process(k: Variable, v: Any):
        if k.is_leaf():
            k.accumulate_derivative(v)
        elif not k.is_constant():
            for k1, v1 in k.chain_rule(d_output=v):
                process(k1, v1)
    
    process(variable, deriv)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
