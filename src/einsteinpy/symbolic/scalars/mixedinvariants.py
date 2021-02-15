from einsteinpy.symbolic.scalars.scalar import BaseRelativityScalar
from einsteinpy.symbolic.tensors.christoffel import ChristoffelSymbols
from einsteinpy.symbolic.tensors.ricci import RicciTensor
from einsteinpy.symbolic.tensors.riemann import RiemannCurvatureTensor
from einsteinpy.symbolic.tensors.weyl import WeylTensor,DualWeylTensor
from sympy import tensorproduct,tensorcontraction,simplify
import numpy as np

class MixedInvariant(BaseRelativityScalar):

    """
    Base class for defining Mixed Invariants
    """

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):

        """
        Get Mixed Invariant calculated from riemann Tensor,method depends on invariant

        Parameters
        ----------
        riemann: ~einsteinpy.symbolic.tensors.riemann.RiemannTensor
            Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor or None
            Corresponding Metric for the First Mixed Invariant.
            Defaults to None.

        """
        pass

    @classmethod
    def from_christoffels(cls, chris, parent_metric=None):
        """
        Get Mixed Invariant calculated from Christoffel symbols

        Parameters
        ----------
        chris : ~einsteinpy.symbolic.tensors.christoffel.ChristoffelSymbols
            Christoffel Symbols
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
        Get Mixed Invariant calculated from Metric Tensor

        Parameters
        ----------
        metric : ~einsteinpy.symbolic.metric.MetricTensor
            Metric Tensor

        """
        ch = ChristoffelSymbols.from_metric(metric)
        return cls.from_christoffels(ch, parent_metric=None)

class FirstMixedInvariant(MixedInvariant):

    """
    Class for defining First Mixed Invariant
    """

    _default = {
        'name':'FirstMixedInvariant'
    }

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):

        """
        Get First Mixed Invariant calculated from riemann Tensor equation given by:

        ..math:: 
                C_{ikl}{}^{j}{}R^{kl}{}R_{j}{}^{i}{}

        Parameters
        ----------
        riemann: ~einsteinpy.symbolic.tensors.riemann.RiemannTensor
            Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor or None
            Corresponding Metric for the First Mixed Invariant.
            Defaults to None.

        """

        #need this form of mixed weyl
        weyl = WeylTensor.from_riemann(riemann).change_config('lllu').tensor()
        #ricci tensor
        ricci = RicciTensor.from_riemann(riemann)
        #need contravariant and mixed forms for ricci
        ricci_mix = ricci.change_config('lu').tensor()
        ricci_con = ricci.change_config('uu').tensor()



        if parent_metric is None:
            parent_metric = riemann.parent_metric

        #take product, contract twice, then take another product and contract twice more. This minimizes computation effort.
        first_mixed = tensorproduct(weyl,ricci_con)

        for i in ((1,4),(1,3)):
            first_mixed = tensorcontraction(first_mixed, i)

        first_mixed = tensorproduct(first_mixed,ricci_mix)
        for i in ((0,3),(0,1)):

            first_mixed = tensorcontraction(first_mixed, i)

        return cls(
            simplify(first_mixed),
            riemann.syms,
            parent_metric=parent_metric,
        )



class SecondMixedInvariant(MixedInvariant):

    """
    Class for defining Second Mixed Invariant
    """
    _default = {
            'name':'SecondMixedInvariant'
        }

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):
        """
        Get Second Mixed Invariant calculated from riemann Tensor equation given by:

        ..math:: 
                -\\star{C}_{ikl}{}^{j}{}R^{kl}{}R_{j}{}^{i}{}

            where \\star{C} is the dual of the weyl tensor

        Parameters
        ----------
        riemann: ~einsteinpy.symbolic.tensors.riemann.RiemannTensor
            Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor or None
            Corresponding Metric for the SecondMixedInvariant.
            Defaults to None.

        """

        #need this form of dual mixed weyl
        weyl_dual = DualWeylTensor.from_riemann(riemann).change_config('lllu').tensor()
        #ricci tensor
        ricci = RicciTensor.from_riemann(riemann)
        #need contravariant and mixed forms for ricci
        ricci_mix = ricci.change_config('lu').tensor()
        ricci_con = ricci.change_config('uu').tensor()



        if parent_metric is None:
            parent_metric = riemann.parent_metric

        #take product, contract twice, then take another product and contract twice more. This minimizes computation effort.
        second_mixed = tensorproduct(-1*weyl_dual,ricci_con)

        for i in ((1,4),(1,3)):
            second_mixed = tensorcontraction(second_mixed, i)

        second_mixed = tensorproduct(second_mixed,ricci_mix)
        for i in ((0,3),(0,1)):

            second_mixed = tensorcontraction(second_mixed, i)

        return cls(
            simplify(second_mixed),
            riemann.syms,
            parent_metric=parent_metric,
        )

class ThirdMixedInvariant(MixedInvariant):

    """
    Class for defining Third Mixed Invariant
    """

    _default = {
        'name':'ThirdMixedInvariant'
    }


    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):
        """
        Get Third Mixed Invariant calculated from riemann Tensor equation given by:

        ..math:: 
                R^{ij}{}R^{kl}{}\\left(C_{oij}{}^{p}{}C_{pkl}{}^{o}{}-\\star{C}_{oij}{}^{p}{}\\star{C}_{pkl}{}^{o}{}\\right)

            where \\star{C} is the dual of the weyl tensor

        Parameters
        ----------
        riemann: ~einsteinpy.symbolic.tensors.riemann.RiemannTensor
            Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor or None
            Corresponding Metric for the ThirdMixedInvariant.
            Defaults to None.

        """
        #weyl tensor
        weyl = WeylTensor.from_riemann(riemann)

        #need this form of dual mixed weyl
        weyl_dual = DualWeylTensor.from_weyltensor(weyl).change_config('lllu').tensor()

        #need this form of mixed weyl
        weyl_mix = weyl.change_config('lllu').tensor()


        #need contravariant form for ricci
        ricci = RicciTensor.from_riemann(riemann).change_config('uu').tensor()

        if parent_metric is None:
            parent_metric = riemann.parent_metric
        
        shape = weyl_mix.shape
        third_mixed = 0
        for t in range(np.product(shape)):
            i,j,k,l = np.unravel_index(t,shape = shape)
            for n in range(np.product(shape[0:2])):
                o,p = np.unravel_index(n,shape = shape[0:2])

                third_mixed += (ricci[i,j]*ricci[k,l]*(weyl_mix[o,i,j,p]*weyl_mix[p,k,l,o] - weyl_dual[o,i,j,p]*weyl_dual[p,k,l,o]))



        return cls(
            simplify(third_mixed),
            riemann.syms,
            parent_metric=parent_metric,
        )


class FourthMixedInvariant(MixedInvariant):

    """
    Class for defining Fourth Mixed Invariant
    """

    _default = {
        'name':'FourthMixedInvariant'
    }

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):
        """
        Get Fourth Mixed Invariant calculated from riemann Tensor equation given by:

        ..math:: 
                -R^{ij}{}R^{kl}{}\\left(\\star{C}_{oij}{}^{p}{}C_{pkl}{}^{o}{}+C_{oij}{}^{p}{}\\star{C}_{pkl}{}^{o}{}\\right)

            where \\star{C} is the dual of the weyl tensor

        Parameters
        ----------
        riemann: ~einsteinpy.symbolic.tensors.riemann.RiemannTensor
            Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor or None
            Corresponding Metric for the FourthMixedInvariant.
            Defaults to None.

        """
        #weyl tensor
        weyl = WeylTensor.from_riemann(riemann)

        #need this form of dual mixed weyl
        weyl_dual = DualWeylTensor.from_weyltensor(weyl).change_config('lllu').tensor()

        #need this form of mixed weyl
        weyl_mix = weyl.change_config('lllu').tensor()


        #need contravariant form for ricci
        ricci = RicciTensor.from_riemann(riemann).change_config('uu').tensor()

        if parent_metric is None:
            parent_metric = riemann.parent_metric
        
        shape = weyl_mix.shape
        fourth_mixed = 0
        for t in range(np.product(shape)):
            i,j,k,l = np.unravel_index(t,shape = shape)
            for n in range(np.product(shape[0:2])):
                o,p = np.unravel_index(n,shape = shape[0:2])

                fourth_mixed += (-1*ricci[i,j]*ricci[k,l]*(weyl_dual[o,i,j,p]*weyl_mix[p,k,l,o] + weyl_mix[o,i,j,p]*weyl_dual[p,k,l,o]))



        return cls(
            simplify(fourth_mixed),
            riemann.syms,
            parent_metric=parent_metric,
        )



class SeventhMixedInvariant(MixedInvariant):

    """
    Class for defining Seventh Mixed Invariant
    """

    _default = {
        'name':'SeventhMixedInvariant'
    }

    @classmethod
    def from_riemann(cls, riemann, parent_metric=None):
        """
        Get Seventh Mixed Invariant calculated from riemann Tensor equation given by:

        ..math:: 
                \\frac{1}{16}R^{ij}{}R^{kl}{}\\left(C_{oijp}{}C^{o}{}_{kl}{}^{p}{}+\\star{C}_{oijp}{}\\star{C}^{o}{}_{kl}{}^{p}{}\\right)

            where \\star{C} is the dual of the weyl tensor

        Parameters
        ----------
        riemann: ~einsteinpy.symbolic.tensors.riemann.RiemannTensor
            Riemann Tensor
        parent_metric : ~einsteinpy.symbolic.tensors.metric.MetricTensor or None
            Corresponding Metric for the SeventhMixedInvariant.
            Defaults to None.

        """
        #weyl tensor
        weyl = WeylTensor.from_riemann(riemann)

        #need dual of weyl
        weyl_dual = DualWeylTensor.from_weyltensor(weyl)

        #need these forms of dual mixed weyl
        weyl_dual2 = weyl_dual.change_config('ullu').tensor()
        weyl_dual1 =weyl_dual.change_config('llll').tensor()

        #need this form of mixed weyl
        weyl_mix1 = weyl.change_config('llll').tensor()
        weyl_mix2 = weyl.change_config('ullu').tensor()


        #need contravariant form for ricci
        ricci = RicciTensor.from_riemann(riemann).change_config('uu').tensor()

        if parent_metric is None:
            parent_metric = riemann.parent_metric
        
        shape = weyl_mix1.shape
        seventh_mixed = 0
        for t in range(np.product(shape)):
            i,j,k,l = np.unravel_index(t,shape = shape)
            for n in range(np.product(shape[0:2])):
                o,p = np.unravel_index(n,shape = shape[0:2])

                seventh_mixed += ((1/16)*ricci[i,j]*ricci[k,l]*(weyl_mix1[o,i,j,p]*weyl_mix2[o,k,l,p] + weyl_dual1[o,i,j,p]*weyl_dual2[o,k,l,p]))



        return cls(
            simplify(seventh_mixed),
            riemann.syms,
            parent_metric=parent_metric,
        )
