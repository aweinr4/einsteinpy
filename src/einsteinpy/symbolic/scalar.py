import numbers
import sympy as sp
from einsteinpy.symbolic import MetricTensor
from sympy.core.function import AppliedUndef, UndefinedFunction

   
class Scalar:
    """
    Base Class for Scalar manipulation
    """

    def __init__(self, expr, name=None):
        """
        Constructor and Initializer

        Parameters
        ----------
        expr : ~sympy.core.expr.Expr or ~ numbers.Number
                Any sympy expression containing only sympy variables or
                a python number
            
        name : str or None
            Name of the scalar.

        Raises
        ------
        TypeError
            Raised when expr is not a singular expression

        """

        if isinstance(expr,sp.core.expr.Expr):
            self.expr = expr
        elif isinstance(expr,numbers.Number):
            self.expr = sp.sympify(expr)
        else:
            raise TypeError("Only numbers and Sympy expressions are supported")
       
        self.name = name
        

  
    def __str__(self):
        """
        Returns a String with a readable representation of the object of class Scalar

        """
        representation = "Scalar"
        if self.name is not None:
            representation = " ".join((representation, self.name))
        representation += "\n"
        representation += self.expr.__str__()
        return representation

    def __repr__(self):
        """
        Returns a String with a representation of the state of the object of class Scalar

        """
        interpretable_representation = self.__class__.__name__
        interpretable_representation += self.expr.__repr__()
        return interpretable_representation


    def subs(self, *args):
        """
        Substitute the variables/expressions in a Scalar with other sympy variables/expressions.

        Parameters
        ----------
        args : one argument or two argument
            - two arguments, e.g foo.subs(old, new)
            - one iterable argument, e.g foo.subs([(old1, new1), (old2, new2)]) for multiple substitutions at once.

        Returns
        -------
        ~einsteinpy.symbolic.scalar.Scalar:
            scalar with substituted values

        """
        return Scalar(self.expression().subs(*args),name = self.name)

    def simplify(self, set_self=True):
        """
        Returns a simplified Scalar

        Parameters
        ----------
        set_self : bool
            Replaces the scalar contained the class with its simplified version, if ``True``.
            Defaults to ``True``.

        Returns
        -------
        ~einsteinpy.symbolic.scalar.Scalar
            Simplified Scalar

        """
        if set_self:
            self.expr = self.expr.simplify()
            return self

        return Scalar(self.expr.simplify(),name = self.name)

    def expression(self):
        return self.expr


class BaseRelativityScalar(Scalar):
    """
    Generic class for defining scalars in General Relativity.
    This would act as a base class for other scalar quantities in GR.

    Attributes
    ----------
    expr : ~sympy.core.expr.Expr or numbers.Number
        Raw sympy expression
    syms : list or tuple
        List of symbols denoting space and time axis of assosiated metric
    dims : int
        dimension of the space-time.
    variables : list
        free variables in the scalar expression other than the variables describing space-time axis.
    functions : list
        Undefined functions in the scalar expression.
    name : str or None
        Name of the scalar. Defaults to "GenericScalar".

    """

    def __init__(
        self,
        expr,
        syms,
        parent_metric=None,
        variables=list(),
        functions=list(),
        name="GenericScalar",
    ):
        """
        Constructor and Initializer

        Parameters
        ----------
        expr : ~sympy.core.expr.Expr or numbers.Number
            Raw sympy expression
        syms : tuple or list
            List of crucial symbols dentoting time-axis and/or spacial axis.
            For example, in case of 4D space-time, the arrangement would look like [t, x1, x2, x3].
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Metric Tensor for some particular space-time which is associated with this Scalar.
        variables : tuple or list or set
            List of symbols used in expressing the scalar,
            other than symbols associated with denoting the space-time axis.
            Calculates in real-time if left blank.
        functions : tuple or list or set
            List of symbolic functions used in epressing the scalar.
            Calculates in real-time if left blank.
        name : str or None
            Name of the Scalar. Defaults to "GenericScalar".

        Raises
        ------
        TypeError
            Raised when expr is not a sympy expression or python number.
        TypeError
            Raised when arguments syms, variables, functions have data type other than list, tuple or set.
        TypeError
            Raised when argument parent_metric does not belong to MetricTensor class and isn't None.

        """
        super(BaseRelativityScalar, self).__init__(expr=expr, name=name)

        if not isinstance(parent_metric,(MetricTensor,type(None))):
            raise TypeError('parent_metric must be of type MetricTensor or nonetype')


        self._parent_metric = parent_metric
        if isinstance(syms, (list, tuple)):
            self.syms = syms
            self.dims = len(self.syms)
        else:
            raise TypeError("syms should be a list or tuple")

        if isinstance(variables, (list, tuple, set)) and isinstance(
            functions, (list, tuple, set)
        ):
            # compute free variables and functions if list if empty
            if not variables:
                self.variables = [
                    v for v in self.expr.free_symbols if v not in self.syms
                ]
                self.variables.sort(key=(lambda var: var.name))
            else:
                self.variables = list(variables)
            if not functions:
                self.functions = [
                    f
                    for f in self.expr.atoms(AppliedUndef).union(
                        self.expr.atoms(UndefinedFunction)
                    )
                ]
            else:
                self.functions = list(functions)

        else:
            raise TypeError(
                "arguments variables and functions should be a list, tuple or set"
            )

    @property
    def parent_metric(self):
        """
        Returns the Metric from which Scalar was derived/associated, if available.
        """
        return self._parent_metric

    def symbols(self):
        """
        Returns the symbols used for defining the time & spacial axis

        Returns
        -------
        tuple
            tuple containing (t,x1,x2,x3) in case of 4D space-time

        """
        return self.syms

 