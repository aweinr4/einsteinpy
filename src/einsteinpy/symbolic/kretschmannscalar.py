from einsteinpy.symbolic import BaseRelativityTensor,RiemannCurvatureTensor,ChristoffelSymbols
from sympy import tensorproduct,tensorcontraction
from einsteinpy.symbolic.helpers import simplify_sympy_array


class KretschmannScalar(BaseRelativityTensor):
    """
    Class for defining Kretschmann Scalar
    """

    def __init__(self, arr, syms, parent_metric=None):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
            Sympy Array, multi-dimensional list containing Sympy Expressions, or Sympy Expressions or int or float scalar
        syms : tuple or list
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Kretschmann Scalar.
            Defaults to None.

        Raises
        ------
        TypeError
            Raised when syms is not a list or tuple

        """
        super(KretschmannScalar, self).__init__(
            arr=arr,
            syms=syms,
            config="",
            parent_metric=parent_metric,
            name="KretschmannScalar",
        )
        self._order = 0

    @property
    def expr(self):
        """
        Retuns the symbolic expression of the Kretschmann Scalar
        """
        val = sum(self.arr)  # sympy not allowing indexing, temporary fix!
        return val

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):
        """
        Get Kretschmann Scalar calculated from Riemann Tensor,equation given by:

        ..math:: 
                R_{ijkl}{}R^{ijkl}{}


        Parameters
        ----------
        RiemannTensor: ~einsteinpy.symbolic.metric.RiemannCurvatureTensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Kretschmann Scalar.
            Defaults to None.

        """

        if not riemann.config == "llll":
            riemann_cov = riemann.change_config(
                newconfig="llll", metric=parent_metric
            ).tensor()
        else:
            riemann_cov = riemann.tensor()

        if not riemann.config == "uuuu":
            riemann_con = riemann.change_config(
                newconfig="uuuu", metric=parent_metric
            ).tensor()
        else:
            riemann_con = riemann.tensor()


        if parent_metric is None:
            parent_metric = riemann.parent_metric

        kretschmann_scalar = tensorproduct(riemann_cov,riemann_con)

        #contracts over each pair accounding for the indices that are gone from last sum
        for i in ((0,4),(0,3),(0,2),(0,1)):
            kretschmann_scalar = tensorcontraction(kretschmann_scalar, i)
        return cls(
            simplify_sympy_array(kretschmann_scalar),
            riemann.syms,
            parent_metric=parent_metric,
        )

    @classmethod
    def from_christoffels(cls, chris, parent_metric=None):
        """
        Get Kretschmann Scalar calculated from Christoffel Tensor

        Parameters
        ----------
        chris : ~einsteinpy.symbolic.christoffel.ChristoffelSymbols
            Christoffel Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Kretschmann Scalar.
            Defaults to None.

        """
        rt = RiemannCurvatureTensor.from_christoffels(
            chris, parent_metric=parent_metric
        )
        return cls.from_riemann(rt)

    @classmethod
    def from_metric(cls, metric):
        """
        Get Kretschmann Scalar calculated from Metric Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.metric.MetricTensor
            Metric Tensor

        """
        ch = ChristoffelSymbols.from_metric(metric)
        return cls.from_christoffels(ch, parent_metric=None)