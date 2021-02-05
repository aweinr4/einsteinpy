from einsteinpy.symbolic import BaseRelativityTensor,RicciTensor,RiemannCurvatureTensor,ChristoffelSymbols,RicciScalar
from sympy import tensorproduct,tensorcontraction,simplify


class SecondRicciInvariant(RicciScalar):
    """
    Class for defining Second Ricci Invariant, use Ricci Scalar as parent because from_christoffels and from_riemann will be same
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
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(SecondRicciInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
        )
        self.name = 'SecondRicciInvariant'


    @classmethod
    def from_riccitensor(cls, riccitensor, parent_metric=None):
        """
        Get Second Ricci Invariant calculated from Ricci Tensor, equation given by: 

        ..math:: 
                R_{ij}{}R^{ij}{}

        Parameters
        ----------
        riccitensor: ~einsteinpy.symbolic.metric.RicciTensor
            Ricci Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        """

        if not riccitensor.config == "ll":
            ricci_cov = riccitensor.change_config(
                newconfig="ll", metric=parent_metric
            ).tensor()
        else:
            ricci_cov = riccitensor.tensor()

        if not riccitensor.config == "uu":
            ricci_con = riccitensor.change_config(
                newconfig="uu", metric=parent_metric
            ).tensor()
        else:
            ricci_con = riccitensor.tensor()

        if parent_metric is None:
            parent_metric = riccitensor.parent_metric

        second_ricci = tensorproduct(ricci_cov,ricci_con)
        for i in ((0,2),(0,1)):
            second_ricci = tensorcontraction(second_ricci, i)

        

        return cls(
            simplify(second_ricci),
            riccitensor.syms,
            parent_metric=parent_metric,
        )

class ThirdRicciInvariant(RicciScalar):
    """
    Class for defining Third Ricci Invariant
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
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(ThirdRicciInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
        )
        self.name = 'ThirdRicciInvariant'


    @classmethod
    def from_riccitensor(cls, riccitensor, parent_metric=None):
        """
        Get Third Ricci Invariant calculated from Ricci Tensor, equation given by: 

        ..math:: 
                R_{i}{}^{j}{}R_{j}{}^{k}{}R_{k}{}^{i}{}

        Parameters
        ----------
        riccitensor: ~einsteinpy.symbolic.metric.RicciTensor
            Ricci Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        """

        #need mixed form of ricci tenosr
        if not riccitensor.config == "lu":
            ricci_mix = riccitensor.change_config(
                newconfig="lu", metric=parent_metric
            ).tensor()
        else:
            ricci_mix = riccitensor.tensor()


        if parent_metric is None:
            parent_metric = riccitensor.parent_metric

        #tensor product of three mixed riccis
        third_ricci = tensorproduct(*[ricci_mix]*3)

        #contracts over each pair accounding for the indices that are gone from last sum
        for i in ((0,5),(0,1),(0,1)):
            third_ricci = tensorcontraction(third_ricci, i)

        

        return cls(
            simplify(third_ricci),
            riccitensor.syms,
            parent_metric=parent_metric,
        )

class FourthRicciInvariant(RicciScalar):
    """
    Class for defining Fourth Ricci Invariant
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
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(FourthRicciInvariant, self).__init__(
            expr=expr,
            syms=syms,
            parent_metric=parent_metric,
        )
        self.name = 'FourthRicciInvariant'


    @classmethod
    def from_riccitensor(cls, riccitensor, parent_metric=None):
        """
        Get Fourth Ricci Invariant calculated from Ricci Tensor, equation given by: 

        ..math:: 
                R_{i}{}^{j}{}R_{j}{}^{k}{}R_{k}{}^{l}{}R_{l}{}^{i}{}

        Parameters
        ----------
        riccitensor: ~einsteinpy.symbolic.metric.RicciTensor
            Ricci Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        """

        #need mixed form of ricci tenosr
        if not riccitensor.config == "lu":
            ricci_mix = riccitensor.change_config(
                newconfig="lu", metric=parent_metric
            ).tensor()
        else:
            ricci_mix = riccitensor.tensor()


        if parent_metric is None:
            parent_metric = riccitensor.parent_metric

        #tensor product of three mixed riccis
        fourth_ricci = tensorproduct(*[ricci_mix]*4)

        #contracts over each pair accounding for the indices that are gone from last sum
        for i in ((0,7),(0,1),(0,1),(0,1)):
            fourth_ricci = tensorcontraction(fourth_ricci, i)

        return cls(
            simplify(fourth_ricci),
            riccitensor.syms,
            parent_metric=parent_metric,
        )