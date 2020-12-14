from objax import TrainVar
from objax.typing import JaxArray
from objax.variable import reduce_mean
from typing import Optional, Callable
from .transforms import Transform, Softplus


class Parameter(TrainVar):
    """
    Base parameter class. This is a simple extension of the `TrainVar` class in Objax that enables parameter transforms
    and, in the future, prior distributions to be placed on the parameter in question.
    """
    def __init__(self,
                 tensor: JaxArray,
                 reduce: Optional[Callable[[JaxArray],
                                           JaxArray]] = reduce_mean,
                 transform: Transform = Softplus()):
        """
        Args:
            tensor: The initial value of the parameter
            reduce: A helper function for parallelisable calls.
            transform: The bijective transformation that should be applied to the parameter.
        """
        super().__init__(transform.forward(tensor), reduce)
        self.fn = transform

    @property
    def untransform(self) -> JaxArray:
        """
        Return the paramerter's transformed valued that exists on constrained R.
        """
        return self.fn.backward(self.value)
