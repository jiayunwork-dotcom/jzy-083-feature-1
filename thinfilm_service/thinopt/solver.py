"""Single-point reflectance / transmittance / absorptance solver.

From the system matrix ``M = [[m11, m12], [m21, m22]]`` and the boundary
admittances ``eta_0`` (incident) and ``eta_s`` (substrate):

    B = m11 + m12 * eta_s          C = m21 + m22 * eta_s
    input admittance  Y = C / B
    amplitude reflection  r = (eta_0 - Y) / (eta_0 + Y)
    amplitude transmission  t = 2 eta_0 / (B * (eta_0 + Y))

    R = |r|^2
    T = Re(eta_s) / Re(eta_0) * |t|^2
    A = 1 - R - T

(For a layer-free system M is the identity, B = 1, and the coefficients
reduce to the textbook single-interface Fresnel formulas.)

The incident medium is required to be lossless (validated upstream), which
keeps the incident flux ``Re(eta_0)`` well defined and guarantees the energy
balance ``R + T = 1`` for lossless stacks and ``R + A + T = 1`` in general.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .matrices import Stack, stack_matrix
from .optics import POL_AVG, POL_P, POL_S

#: Residuals below this magnitude are floating-point noise, not physics.
_NOISE_FLOOR = 1e-12


@dataclass(frozen=True)
class PolarizationResult:
    """Energy coefficients for one polarisation at one (wavelength, angle)."""

    reflectance: float
    transmittance: float
    absorptance: float
    r: complex  # amplitude reflection coefficient
    t: complex  # amplitude transmission coefficient


@dataclass(frozen=True)
class PointResult:
    """Full single-point result; ``s``/``p`` hold the per-polarisation values."""

    wavelength: float
    angle_deg: float
    s: PolarizationResult
    p: PolarizationResult

    def averaged(self) -> PolarizationResult:
        """Unpolarised light: plain mean of the s and p energy coefficients."""
        s, p = self.s, self.p
        return PolarizationResult(
            reflectance=0.5 * (s.reflectance + p.reflectance),
            transmittance=0.5 * (s.transmittance + p.transmittance),
            absorptance=0.5 * (s.absorptance + p.absorptance),
            r=0.5 * (s.r + p.r),
            t=0.5 * (s.t + p.t),
        )


def _clean(value: float) -> float:
    """Zero out sign-flipped floating-point noise (e.g. A = -3e-17)."""
    if abs(value) < _NOISE_FLOOR:
        return 0.0
    return value


def solve_polarization(
    stack: Stack,
    wavelength: float,
    angle_deg: float,
    polarization: str,
) -> PolarizationResult:
    """R/T/A for one polarisation at one wavelength and angle of incidence."""
    matrix, eta_0, eta_s = stack_matrix(stack, wavelength, math.radians(angle_deg), polarization)
    m11, m12 = matrix[0, 0], matrix[0, 1]
    m21, m22 = matrix[1, 0], matrix[1, 1]

    b = m11 + m12 * eta_s
    c = m21 + m22 * eta_s
    y = c / b
    r = (eta_0 - y) / (eta_0 + y)
    t = (2.0 * eta_0) / (b * (eta_0 + y))

    reflectance = float(abs(r) ** 2)
    transmittance = float((eta_s.real / eta_0.real) * abs(t) ** 2)
    absorptance = 1.0 - reflectance - transmittance
    return PolarizationResult(
        reflectance=_clean(reflectance),
        transmittance=_clean(transmittance),
        absorptance=_clean(absorptance),
        r=complex(r),
        t=complex(t),
    )


def solve_point(stack: Stack, wavelength: float, angle_deg: float) -> PointResult:
    """Solve both polarisations at one wavelength and angle of incidence."""
    return PointResult(
        wavelength=wavelength,
        angle_deg=angle_deg,
        s=solve_polarization(stack, wavelength, angle_deg, POL_S),
        p=solve_polarization(stack, wavelength, angle_deg, POL_P),
    )


def select_polarization(point: PointResult, polarization: str) -> PolarizationResult:
    """Pick one polarisation view of a :class:`PointResult` (``s``/``p``/``avg``)."""
    if polarization == POL_S:
        return point.s
    if polarization == POL_P:
        return point.p
    if polarization == POL_AVG:
        return point.averaged()
    raise ValueError(f"unknown polarisation: {polarization!r}")
