from einsteinpy.symbolic.scalars.scalar import BaseRelativityScalar
from einsteinpy.symbolic.tensors.christoffel import ChristoffelSymbols
from einsteinpy.symbolic.tensors.riemann import RiemannCurvatureTensor
from einsteinpy.symbolic.tensors.weyl import WeylTensor,DualWeylTensor
from sympy import tensorproduct,tensorcontraction,simplify

class FirstWeylInvariant(BaseRelativityScalar):

    """
    Class for defining First Weyl Invariant
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
            Corresponding Metric for the First Weyl Invariant.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(FirstWeylInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
            name="FirstWeylInvariant",
        )


    @classmethod
    def from_weyltensor(cls, weyl, parent_metric=None):
        """
        Get First Weyl Invariant calculated from Ricci Tensor equation given by:

        ..math:: 
                C_{ij}{}^{kl}{}C_{kl}{}^{ij}{}

        Parameters
        ----------
        weyltensor: ~einsteinpy.symbolic.tensors.weyl.WeylTensor
            Weyl Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the First Weyl Invariant.
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
        Get First Weyl Invariant calculated from Riemann Tensor

        Parameters
        ----------
        riemann : ~einsteinpy.symbolic.riemann.RiemannCurvatureTensor
           Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the First Weyl Invariant.
            Defaults to None.

        """

        weyl = WeylTensor.from_riemann(riemann, parent_metric=parent_metric)
        return cls.from_weyltensor(weyl)

    @classmethod
    def from_christoffels(cls, chris, parent_metric=None):
        """
        Get First Weyl Invariant calculated from Christoffel Tensor

        Parameters
        ----------
        chris : ~einsteinpy.symbolic.christoffel.ChristoffelSymbols
            Christoffel Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the First Weyl Invariant.
            Defaults to None.

        """
        rt = RiemannCurvatureTensor.from_christoffels(
            chris, parent_metric=parent_metric
        )
        return cls.from_riemann(rt)

    @classmethod
    def from_metric(cls, metric):
        
        """
        Get First Weyl Invariant calculated from Metric Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.metric.MetricTensor
            Metric Tensor

        """
        ch = ChristoffelSymbols.from_metric(metric)
        return cls.from_christoffels(ch, parent_metric=None)

class SecondWeylInvariant(FirstWeylInvariant):
    """
    Class for defining Second Weyl Invariant
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
            Corresponding Metric for the Second Weyl Invariant.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(SecondWeylInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
        )
        self.name="SecondWeylInvariant"

    @classmethod
    def from_weyltensor(cls, weyl, parent_metric=None):
        """
        Get Second Weyl Invariant calculated from weyl Tensor equation given by:

        ..math:: 
                -C_{ij}{}^{kl}{}\\star{C}_{kl}{}^{ij}{}
        
        where \\star{C} is the dual of the weyl tensor

        Parameters
        ----------
        weyltensor: ~einsteinpy.symbolic.tensors.weyl.WeylTensor
            Weyl Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Second Weyl Invariant.
            Defaults to None.

        """

        weyl_dual = DualWeylTensor.from_weyltensor(weyl).change_config('lluu')

        if not weyl.config == "lluu":
            weyl = weyl.change_config(
                newconfig="lluu", metric=parent_metric
            )

        if parent_metric is None:
            parent_metric = weyl.parent_metric
        
        
        second_weyl = tensorproduct(-1*weyl.tensor(),weyl_dual.tensor())
        for i in ((0,6),(0,5),(0,2),(0,1)):

            second_weyl = tensorcontraction(second_weyl, i)
        
        
        return cls(
            simplify(second_weyl),
            weyl.syms,
            parent_metric=parent_metric,
        )

class ThirdWeylInvariant(FirstWeylInvariant):

    """
    Class for defining Third Weyl Invariant
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
            Corresponding Metric for the Third Weyl Invariant.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(ThirdWeylInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
        )
        self.name="ThirdWeylInvariant"

    @classmethod
    def from_weyltensor(cls, weyl, parent_metric=None):
        """
        Get Third Weyl Invariant calculated from weyl Tensor equation given by:

        ..math:: 
                C_{ij}{}^{kl}{}C_{kl}{}^{op}{}C_{op}{}^{}{ij}

        Parameters
        ----------
        weyltensor: ~einsteinpy.symbolic.tensors.weyl.WeylTensor
            Weyl Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Third Weyl Invariant.
            Defaults to None.

        """

        if not weyl.config == "lluu":
            weyl = weyl.change_config(
                newconfig="lluu", metric=parent_metric
            )
        if parent_metric is None:
            parent_metric = weyl.parent_metric
        
        #take product and contract two indices then take another product and contract two more, this avoids getting too large
        third_weyl = tensorproduct(*[weyl.tensor()]*2)
        for i in ((2,4),(2,3)):

            third_weyl = tensorcontraction(third_weyl, i)
        
        third_weyl = tensorproduct(third_weyl,weyl.tensor())
        for i in ((0,6),(0,5),(0,2),(0,1)):

            third_weyl = tensorcontraction(third_weyl, i)

        
        return cls(
            simplify(third_weyl),
            weyl.syms,
            parent_metric=parent_metric,
        )

class FourthWeylInvariant(FirstWeylInvariant):
    """
    Class for defining Fourth Weyl Invariant
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
            Corresponding Metric for the Fourth Weyl Invariant.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(FourthWeylInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
        )
        self.name="FourthWeylInvariant"

    @classmethod
    def from_weyltensor(cls, weyl, parent_metric=None):
        """
        Get Fourth Weyl Invariant calculated from weyl Tensor equation given by:

        ..math:: 
                -C_{ij}{}^{kl}{}\\star{C}_{kl}{}^{op}{}C_{op}{}^{ij}{}
        
        where \\star{C} is the dual of the weyl tensor

        Parameters
        ----------
        weyltensor: ~einsteinpy.symbolic.tensors.weyl.WeylTensor
            Weyl Tensor
        parent_metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor or None
            Corresponding Metric for the Fourth Weyl Invariant.
            Defaults to None.

        """

        weyl_dual = DualWeylTensor.from_weyltensor(weyl).change_config('lluu')

        if not weyl.config == "lluu":
            weyl = weyl.change_config(
                newconfig="lluu", metric=parent_metric
            )

        if parent_metric is None:
            parent_metric = weyl.parent_metric
        
        '''take first product then contract two indices and then take another product. 
         This prevents large tensor from slowing computation'''
        fourth_weyl = tensorproduct(-1*weyl_dual.tensor(),weyl.tensor())
        for i in ((2,4),(2,3)):

            fourth_weyl = tensorcontraction(fourth_weyl, i)

        fourth_weyl = tensorproduct(weyl.tensor(),fourth_weyl)
        for i in ((0,6),(0,5),(0,2),(0,1)):

            fourth_weyl = tensorcontraction(fourth_weyl, i)

        
        return cls(
            simplify(fourth_weyl),
            weyl.syms,
            parent_metric=parent_metric,
        )