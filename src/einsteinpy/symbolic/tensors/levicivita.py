import sympy as sp
import numpy as np
from einsteinpy.symbolic.helpers import _change_name,perm_parity
from einsteinpy.symbolic.tensors.tensor import BaseRelativityTensor,_change_config



class LeviCivitaSymbols(BaseRelativityTensor):
    """
    Class to define the Levi-Civita Symbols based on the metric
    """

    def __init__(self, arr, syms, config="llll", name="LeviCivitaSymbols",parent_metric=None):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
            Sympy Array or multi-dimensional list containing Sympy Expressions
        syms : tuple or list
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        config : str
            Configuration of contravariant and covariant indices in tensor. 'u' for upper and 'l' for lower indices. Defaults to 'll'.
        name : str

        Raises
        ------
        TypeError
            Raised when arr is not a list or sympy Array
        TypeError
            syms is not a list or tuple
        ValueError
            config has more or less than 4 indices

        """
        super(LeviCivitaSymbols, self).__init__(
            arr=arr, syms=syms, config=config, parent_metric=parent_metric, name=name
        )
        self._order = 4
        self._invmetric = None
        if not len(config) == self._order:
            raise ValueError("config should be of length {}".format(self._order))

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
        ~einsteinpy.symbolic.tensors.levicivita.LeviCivitaSymbols
            New tensor with new configuration. Defaults to 'llll'

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
        new_obj = LeviCivitaSymbols(
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
        Get Levi Civita symbols calculated from a metric tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor
            Space-time Metric from which Levi Civita Symbols are to be calculated
            
        Raises
        ------
        TypeError
            Raised when number of coordinates is not 4.

        """
        dims = metric.dims
        if dims !=4:
            raise TypeError('currently the Levi Civita Symbols are only supported for a 4-Dimensional coordinate basis')
        tmplist = []
        mat, syms = metric.lower_config().tensor(), metric.symbols()
        # need sqrt of metric determinant 
        mat_det = sp.sqrt(abs(mat.tomatrix().det())).simplify()
        '''uses permutation parity to craft levi civita tensor and multiplies each 
        index by determinant of metric'''
        for i in range(256):
            indx = [*np.unravel_index(i,shape = [4]*4)]
            tmplist.append(mat_det*perm_parity(indx))
          
        #reshape the flat list
        tmplist = sp.Array(tmplist,[4]*4)
        
        return cls(tmplist, syms, config="llll", parent_metric=metric)
