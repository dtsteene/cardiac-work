"""Apply the pre-99e78f0 fenicsx-pulse active-stress formulation as a monkey
patch on `pulse.active_model.ActiveModel`. Importing this module rewires
active.S and active.P to evaluate on the full right Cauchy-Green tensor C
instead of its isochoric part Cdev = J^(-2/3) C.

Used by the deviatoric-stress counterfactual experiment in the
Validating-the-Mechanical-Reference chapter to reproduce the stress magnitudes
the pre-fix pipeline produced on the lowest-pressure sweep case.
"""
from __future__ import annotations

import ufl
from pulse.active_model import ActiveModel


def _legacy_S(self, C, dev=False):
    return 2.0 * ufl.diff(self.strain_energy(C), C)


def _legacy_P(self, F, dev=False):
    C = F.T * F
    return ufl.diff(self.strain_energy(C), F)


ActiveModel.S = _legacy_S
ActiveModel.P = _legacy_P
