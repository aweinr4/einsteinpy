from einsteinpy.symbolic.scalars.scalar import BaseRelativityScalar
from einsteinpy.symbolic.tensors.riemann import RiemannCurvatureTensor
from einsteinpy.symbolic.tensors.ricci import RicciTensor
from einsteinpy.symbolic.tensors.christoffel import ChristoffelSymbols
from sympy import tensorproduct,tensorcontraction,simplify

class RicciScalar(BaseRelativityScalar):
    """
    Class for defining Ricci Scalar
    """

    _default = {
        'name':'RicciScalar'
    }


    @classmethod
    def from_riccitensor(cls, riccitensor, parent_metric=None):
        """
        Get Ricci Scalar calculated from Ricci Tensor

        Parameters
        ----------
        riccitensor: ~einsteinpy.symbolic.metric.RicciTensor
            Ricci Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        """

        if not riccitensor.config == "ul":
            riccitensor = riccitensor.change_config(
                newconfig="ul", metric=parent_metric
            )
        if parent_metric is None:
            parent_metric = riccitensor.parent_metric
        ricci_scalar = tensorcontraction(riccitensor.tensor(), (0, 1))


        return cls(
            simplify(ricci_scalar),
            riccitensor.syms,
            parent_metric=parent_metric,
        )

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):
        """
        Get Ricci Scalar calculated from Riemann Tensor

        Parameters
        ----------
        riemann : ~einsteinpy.symbolic.riemann.RiemannCurvatureTensor
           Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        """

        cg = RicciTensor.from_riemann(riemann, parent_metric=parent_metric)
        return cls.from_riccitensor(cg)

    @classmethod
    def from_christoffels(cls, chris, parent_metric=None):
        """
        Get Ricci Scalar calculated from Christoffel Tensor

        Parameters
        ----------
        chris : ~einsteinpy.symbolic.christoffel.ChristoffelSymbols
            Christoffel Tensor
        parent_metric : ~einsteinpy.symbolic.metric.MetricTensor or None
            Corresponding Metric for the Ricci Scalar.
            Defaults to None.

        """
        rt = RiemannCurvatureTensor.from_christoffels(
            chris, parent_metric=parent_metric
        )
        return cls.from_riemann(rt)

    @classmethod
    def from_metric(cls, metric):
        """
        Get Ricci Scalar calculated from Metric Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.metric.MetricTensor
            Metric Tensor

        """
        ch = ChristoffelSymbols.from_metric(metric)
        return cls.from_christoffels(ch, parent_metric=None)

class SecondRicciInvariant(RicciScalar):
    """
    Class for defining Second Ricci Invariant, use Ricci Scalar as parent because from_christoffels and from_riemann will be same
    """
    _default = {
        'name':'SecondRicciInvariant'
    }


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

    _default = {
        'name':'ThirdRicciInvariant'
    }


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

    _default = {
        'name':'FourthRicciInvariant'
    }

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