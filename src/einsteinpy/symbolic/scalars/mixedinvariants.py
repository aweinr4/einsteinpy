from einsteinpy.symbolic.scalars.scalar import BaseRelativityScalar
from einsteinpy.symbolic.tensors.christoffel import ChristoffelSymbols
from einsteinpy.symbolic.tensors.riemann import RiemannCurvatureTensor
from einsteinpy.symbolic.tensors.weyl import WeylTensor,DualWeylTensor
from sympy import tensorproduct,tensorcontraction,simplify

class FirstMixedInvariant(BaseRelativityScalar):

    """
    Class for defining First Mixed Invariant
    """

    def __init__(self, expr, syms, parent_metric=None):
        """
        Constructor and Initializer

        Parameters
        ----------
        expr : ~sympy.core.expr.Expr or numbers.Number
            Raw sympy expression
        syms : tuple or list
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the First Mixed Invariant.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(FirstMixedInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
            name="FirstMixedInvariant",
        )


    @classmethod
    def from_weyltensor(cls, weyl, parent_metric=None):
        """
        Get First Mixed Invariant calculated from Ricci Tensor equation given by:

        ..math:: 
                C_{ij}{}^{kl}{}C_{kl}{}^{ij}{}

        Parameters
        ----------
        weyltensor: ~einsteinpy.symbolic.tensors.weyl.WeylTensor
            Weyl Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the First Mixed Invariant.
            Defaults to None.

        """

        if not weyl.config == "lluu":
            weyl = weyl.change_config(
                newconfig="lluu", metric=parent_metric
            )
        if parent_metric is None:
            parent_metric = weyl.parent_metric
        
        first_weyl = tensorproduct(*[weyl.tensor()]*2)
        for i in ((0,6),(0,5),(0,2),(0,1)):

            first_weyl = tensorcontraction(first_weyl, i)
        return cls(
            simplify(first_weyl),
            weyl.syms,
            parent_metric=parent_metric,
        )

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):
        """
        Get First Mixed Invariant calculated from Riemann Tensor

        Parameters
        ----------
        riemann : ~einsteinpy.symbolic.riemann.RiemannCurvatureTensor
           Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the First Mixed Invariant.
            Defaults to None.

        """

        weyl = WeylTensor.from_riemann(riemann, parent_metric=parent_metric)
        return cls.from_weyltensor(weyl)

    @classmethod
    def from_christoffels(cls, chris, parent_metric=None):
        """
        Get First Mixed Invariant calculated from Christoffel Tensor

        Parameters
        ----------
        chris : ~einsteinpy.symbolic.christoffel.ChristoffelSymbols
            Christoffel Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the First Mixed Invariant.
            Defaults to None.

        """
        rt = RiemannCurvatureTensor.from_christoffels(
            chris, parent_metric=parent_metric
        )
        return cls.from_riemann(rt)

    @classmethod
    def from_metric(cls, metric):
        
        """
        Get First Mixed Invariant calculated from Metric Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.metric.MetricTensor
            Metric Tensor

        """
        ch = ChristoffelSymbols.from_metric(metric)
        return cls.from_christoffels(ch, parent_metric=None)