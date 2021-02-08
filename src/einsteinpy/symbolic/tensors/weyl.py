from mpmath.functions.functions import im
import numpy as np
import sympy

from einsteinpy.symbolic.helpers import _change_name
from einsteinpy.symbolic.tensors.riemann import RiemannCurvatureTensor
from einsteinpy.symbolic.tensors.ricci import RicciTensor
from einsteinpy.symbolic.scalars.ricciinvariants import RicciScalar
from einsteinpy.symbolic.tensors.tensor import  _change_config,BaseRelativityTensor
from einsteinpy.symbolic.tensors.levicivita import LeviCivitaSymbols
from sympy import tensorcontraction,tensorproduct


class WeylTensor(BaseRelativityTensor):
    """
    Class for defining Weyl Tensor
    """

    def __init__(self, arr, syms, config="ulll", parent_metric=None, name="WeylTensor"):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
            Sympy Array or multi-dimensional list containing Sympy Expressions
        syms : tuple or list
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        config : str
            Configuration of contravariant and covariant indices in tensor. 'u' for upper and 'l' for lower indices. Defaults to 'ulll'.
        parent_metric : ~einsteinpy.symbolic.metric.WeylTensor
            Corresponding Metric for the Weyl Tensor. Defaults to None.
        name : str
            Name of the Tensor. Defaults to "WeylTensor"

        Raises
        ------
        TypeError
            Raised when arr is not a list or sympy Array
        TypeError
            syms is not a list or tuple
        ValueError
            config has more or less than 4 indices

        """
        super(WeylTensor, self).__init__(
            arr=arr, syms=syms, config=config, parent_metric=parent_metric, name=name
        )
        self._order = 4
        if not len(config) == self._order:
            raise ValueError("config should be of length {}".format(self._order))

    @classmethod
    def from_riemann(cls, riemann,parent_metric = None):
        """
        Get Weyl tensor calculated from a riemann tensor

        Parameters
        ----------
        riemann : ~einsteinpy.symbolic.metric.RiemannCurvatureTensor

        Raises
        ------
        ValueError
            Raised when the dimension of the tensor is less than 3

        """
        if parent_metric is None:
            metric = riemann.parent_metric
        else:
            metric = parent_metric
        if metric.dims > 3:
            g = metric.lower_config()
            t_ricci = RicciTensor.from_riemann(riemann, parent_metric=None)

            # Riemann Tensor with covariant indices is needed
            if not riemann.config == 'llll':
                riemann = riemann.change_config("llll", metric=None)
            
            r_scalar = RicciScalar.from_riccitensor(t_ricci, parent_metric=None)
            dims = g.dims
            # Indexing for resultant Weyl Tensor is iklm
            C = np.zeros(shape=(dims, dims, dims, dims), dtype=int).tolist()
            for t in range(dims ** 4):
                i, k, l, m = (
                    t % dims,
                    (int(t / dims)) % (dims),
                    (int(t / (dims ** 2))) % (dims),
                    (int(t / (dims ** 3))) % (dims),
                )
                C[i][k][l][m] = riemann[i, k, l, m] + (
                    (
                        (
                            t_ricci[i, m] * g[k, l]
                            - t_ricci[i, l] * g[k, m]
                            + t_ricci[k, l] * g[i, m]
                            - t_ricci[k, m] * g[i, l]
                        )
                        / (dims - 2)
                    )
                    + (
                        r_scalar.expr
                        * (g[i, l] * g[k, m] - g[i, m] * g[k, l])
                        / ((dims - 1) * (dims - 2))
                    )
                )
            C = sympy.simplify(sympy.Array(C))
            return cls(C, riemann.syms, config="llll", parent_metric=metric)
        if metric.dims == 3:
            return cls(
                sympy.Array(np.zeros((3, 3, 3, 3), dtype=int)),
                riemann.syms,
                config="llll",
                parent_metric=riemann.parent_metric,
            )
        raise ValueError("Dimension of the space/space-time should be 3 or more")

    def change_config(self, newconfig="llll", metric=None):
        """
        Changes the index configuration(contravariant/covariant)

        Parameters
        ----------
        newconfig : str
            Specify the new configuration. Defaults to 'llll'
        metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Parent metric tensor for changing indices.
            Already assumes the value of the metric tensor from which it was initialized if passed with None.
            Compulsory if not initialized with 'from_metric'. Defaults to None.

        Returns
        -------
        ~einsteinpy.symbolic.weyl.WeylTensor
            New tensor with new configuration. Configuration defaults to 'llll'

        Raises
        ------
        Exception
            Raised when a parent metric could not be found.

        """
        if metric is None:
            metric = self._parent_metric
        if metric is None:
            raise Exception("Parent Metric not found, can't do configuration change")
        new_tensor = _change_config(self, metric, newconfig)
        new_obj = WeylTensor(
            new_tensor,
            self.syms,
            config=newconfig,
            parent_metric=metric,
            name=_change_name(self.name, context="__" + newconfig),
        )
        return new_obj

    @classmethod
    def from_metric(cls, metric):
        """
        Get Weyl tensor calculated from Metric Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.metric.MetricTensor
            Metric Tensor

        """
        riemann = RiemannCurvatureTensor.from_metric(metric)
        return cls.from_riemann(riemann, parent_metric=None)

    def lorentz_transform(self, transformation_matrix):
        """
        Performs a Lorentz transform on the tensor.

        Parameters
        ----------
            transformation_matrix : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
                Sympy Array or multi-dimensional list containing Sympy Expressions

        Returns
        -------
            ~einsteinpy.symbolic.weyl.WeylTensor
                lorentz transformed tensor(or vector)

        """
        t = super(WeylTensor, self).lorentz_transform(transformation_matrix)
        return WeylTensor(
            t.tensor(),
            syms=self.syms,
            config=self._config,
            parent_metric=None,
            name=_change_name(self.name, context="__lt"),
        )


class DualWeylTensor(BaseRelativityTensor):
    """
    Class for defining dual of Weyl Tensor
    """

    def __init__(self, arr, syms, config="llll", parent_metric=None, name="DualWeylTensor"):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
            Sympy Array or multi-dimensional list containing Sympy Expressions
        syms : tuple or list
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        config : str
            Configuration of contravariant and covariant indices in tensor. 'u' for upper and 'l' for lower indices. Defaults to 'ulll'.
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor
            Corresponding Metric for the Weyl Tensor. Defaults to None.
        name : str
            Name of the Tensor. Defaults to "DualWeylTensor"

        Raises
        ------
        TypeError
            Raised when arr is not a list or sympy Array
        TypeError
            syms is not a list or tuple
        ValueError
            config has more or less than 4 indices

        """
        super(DualWeylTensor, self).__init__(
            arr=arr, syms=syms, config=config, parent_metric=parent_metric, name=name
        )
        self._order = 4
        if not len(config) == self._order:
            raise ValueError("config should be of length {}".format(self._order))

    @classmethod
    def from_weyltensor(cls, weyl,parent_metric = None):
        """
        Get dual Weyl tensor calculated from a weyl tensor

        Parameters
        ----------
        weyl : ~einsteinpy.symbolic.tensors.weyl.WeylTensor

        Raises
        ------
        ValueError
            Raised when the dimension of the tensor is not 4

        """
        if parent_metric is None:
            metric = weyl.parent_metric
        else:
            metric = parent_metric
        if metric.dims == 4:
            
            # need levi civita symbols
            levi_c = LeviCivitaSymbols.from_metric(metric)

            # weyl with mixed indices is needed
            if not weyl.config == 'uull':
                weyl = weyl.change_config("uull")

            weyl_dual = tensorproduct(0.5*levi_c.tensor(),weyl.tensor())

            for i in ((2,4),(2,3)):
                weyl_dual = tensorcontraction(weyl_dual,i)

            
            return cls(weyl_dual, weyl.syms, config="llll", parent_metric=metric)
        else:
            raise ValueError("Dimension of the space/space-time should be 4")

    def change_config(self, newconfig="llll", metric=None):
        """
        Changes the index configuration(contravariant/covariant)

        Parameters
        ----------
        newconfig : str
            Specify the new configuration. Defaults to 'llll'
        metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Parent metric tensor for changing indices.
            Already assumes the value of the metric tensor from which it was initialized if passed with None.
            Compulsory if not initialized with 'from_metric'. Defaults to None.

        Returns
        -------
        ~einsteinpy.symbolic.weyl.DualWeylTensor
            New tensor with new configuration. Configuration defaults to 'llll'

        Raises
        ------
        Exception
            Raised when a parent metric could not be found.

        """
        if metric is None:
            metric = self._parent_metric
        if metric is None:
            raise Exception("Parent Metric not found, can't do configuration change")
        new_tensor = _change_config(self, metric, newconfig)
        new_obj = DualWeylTensor(
            new_tensor,
            self.syms,
            config=newconfig,
            parent_metric=metric,
            name=_change_name(self.name, context="__" + newconfig),
        )
        return new_obj


    @classmethod
    def from_riemann(cls, riemann,parent_metric = None):
        """
        Get dual Weyl tensor calculated from riemann Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor
            Metric Tensor

        """
        weyl = WeylTensor.from_metric(riemann)
        return cls.from_weyltensor(weyl, parent_metric=parent_metric)

    
    @classmethod
    def from_metric(cls, metric):
        """
        Get dual Weyl tensor calculated from Metric Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor
            Metric Tensor

        """
        riemann = RiemannCurvatureTensor.from_metric(metric)
        return cls.from_riemann(riemann, parent_metric=None)

    def lorentz_transform(self, transformation_matrix):
        """
        Performs a Lorentz transform on the tensor.

        Parameters
        ----------
            transformation_matrix : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
                Sympy Array or multi-dimensional list containing Sympy Expressions

        Returns
        -------
            ~einsteinpy.symbolic.weyl.DualWeylTensor
                lorentz transformed tensor(or vector)

        """
        t = super(DualWeylTensor, self).lorentz_transform(transformation_matrix)
        return DualWeylTensor(
            t.tensor(),
            syms=self.syms,
            config=self._config,
            parent_metric=None,
            name=_change_name(self.name, context="__lt"),
        )
