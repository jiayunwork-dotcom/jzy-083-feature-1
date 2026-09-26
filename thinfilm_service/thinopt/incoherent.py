"""Finite-thickness substrate: the incoherent back-surface channel.

A real sample is a slab, not a half-space.  Light that crosses the coated
front face meets a second, usually bare, interface — substrate to back-side
medium — and part of it returns to the detector.  When the front coating is
a good anti-reflection stack (front reflectance pushed below ~1 %), the
bare back interface (~4 % for glass in air) dominates the measured
reflectance, so a front-only model sits permanently below the instrument
reading.  This module adds the missing channel.

Two calibers, side by side:

* ``front_only`` — the coherent front stack on a semi-infinite substrate,
  i.e. exactly what :func:`thinopt.solver.solve_polarization` reports.  The
  thin films are far below the coherence length, so they stay fully
  coherent and are solved by the very same characteristic-matrix code
  (:func:`thinopt.matrices.stack_matrix` +
  :func:`thinopt.solver.coefficients_from_matrix`).
* ``with_back`` — the whole finite sample.  The substrate (typically a
  fraction of a millimetre) is far thicker than the coherence length, so
  the phase between successive round trips inside it is washed out: the
  multiple reflections are summed as *intensities*, never as amplitudes.

Per polarisation, with power coefficients

    R_f, T_f    front stack, incidence from the ambient side
    R_f', T_f'  front stack, incidence from the substrate side
    tau         single-pass substrate transmittance (Beer-Lambert)
    R_b, T_b    bare back interface (substrate -> back-side medium)

the round-trip paths form a geometric series that is summed in closed form
(not truncated):

    R_total = R_f + T_f T_f' R_b tau^2 / (1 - R_f' R_b tau^2)
    T_total = T_f T_b tau          / (1 - R_f' R_b tau^2)
    A_total = 1 - R_total - T_total

The single-pass factor reuses the thin-film phase definition,
``tau = |exp(i delta_s)|^2 = exp(-2 Im(delta_s))`` with
``delta_s = (2 pi / lambda) N_s d cos(theta_s)``; for a lossless substrate
``tau = 1`` exactly.  When the back-side medium equals the substrate the
back interface vanishes (``R_b = 0``) and the totals collapse exactly onto
the front-only coefficients — the semi-infinite-substrate limit, locked in
by tests.  All functions are pure: every request computes on its own stack
and intermediates, so concurrent requests never interfere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .matrices import Stack, stack_matrix
from .optics import (
    POL_AVG,
    POL_P,
    POL_S,
    cos_theta_in_layer,
    effective_admittance,
    phase_thickness,
)
from .solver import coefficients_from_matrix

#: Same noise-floor convention as thinopt.solver: residuals below this are
#: floating-point noise (e.g. A = -3e-17 in a lossless cascade), not physics.
_NOISE_FLOOR = 1e-12


def _clean(value: float) -> float:
    if abs(value) < _NOISE_FLOOR:
        return 0.0
    return value


@dataclass(frozen=True)
class EnergyCoefficients:
    """Measurable power coefficients (one polarisation, one caliber)."""

    reflectance: float
    transmittance: float
    absorptance: float


@dataclass(frozen=True)
class SamplePolarizationResult:
    """Both calibers for one polarisation, plus the cascade ingredients.

    ``front_only`` is the coherent semi-infinite-substrate caliber;
    ``with_back`` adds the finite substrate and its back interface
    incoherently.  The remaining fields expose the ingredients so callers
    can see exactly how much the back surface contributes.
    """

    front_only: EnergyCoefficients
    with_back: EnergyCoefficients
    single_pass_transmittance: float  # tau, one crossing of the substrate
    back_interface_reflectance: float  # R_b, bare substrate -> back-medium interface
    back_interface_transmittance: float  # T_b
    front_reverse_reflectance: float  # R_f', front stack seen from the substrate
    front_reverse_transmittance: float  # T_f'


@dataclass(frozen=True)
class SampleResult:
    """Whole-sample result; ``s``/``p`` hold the per-polarisation values."""

    wavelength: float
    angle_deg: float
    substrate_thickness: float
    back_index: complex
    s: SamplePolarizationResult
    p: SamplePolarizationResult

    def averaged(self) -> SamplePolarizationResult:
        """Unpolarised light: plain mean of the s and p values."""
        s, p = self.s, self.p
        return SamplePolarizationResult(
            front_only=_mean_energy(s.front_only, p.front_only),
            with_back=_mean_energy(s.with_back, p.with_back),
            single_pass_transmittance=0.5
            * (s.single_pass_transmittance + p.single_pass_transmittance),
            back_interface_reflectance=0.5
            * (s.back_interface_reflectance + p.back_interface_reflectance),
            back_interface_transmittance=0.5
            * (s.back_interface_transmittance + p.back_interface_transmittance),
            front_reverse_reflectance=0.5
            * (s.front_reverse_reflectance + p.front_reverse_reflectance),
            front_reverse_transmittance=0.5
            * (s.front_reverse_transmittance + p.front_reverse_transmittance),
        )


def _mean_energy(a: EnergyCoefficients, b: EnergyCoefficients) -> EnergyCoefficients:
    return EnergyCoefficients(
        reflectance=0.5 * (a.reflectance + b.reflectance),
        transmittance=0.5 * (a.transmittance + b.transmittance),
        absorptance=0.5 * (a.absorptance + b.absorptance),
    )


def single_pass_transmittance(
    index: complex, cos_theta: complex, thickness: float, wavelength: float
) -> float:
    """Intensity attenuation over one substrate crossing.

    ``tau = |exp(i delta)|^2 = exp(-2 Im(delta))`` with the same phase
    thickness ``delta = (2 pi / lambda) N d cos(theta)`` the thin films use.
    The branch choice in :func:`thinopt.optics.cos_theta_in_layer` guarantees
    ``Im(N cos) >= 0`` for passive media, so ``0 <= tau <= 1``; a lossless
    substrate gives ``tau = 1`` exactly.  At normal incidence this reduces
    to Beer-Lambert, ``exp(-4 pi k d / lambda)``; at oblique incidence the
    ``cos(theta)`` factor lengthens the path, as it does for the films.
    """
    delta = phase_thickness(index, thickness, cos_theta, wavelength)
    return math.exp(-2.0 * delta.imag)


def reverse_coefficients(
    matrix: np.ndarray, eta_0: complex, eta_s: complex
) -> tuple[float, float]:
    """R'/T' of the *same* system matrix for light incident from the substrate.

    The matrix relation ``[E0; H0] = M [Es; Hs]`` is direction-neutral.
    With a wave of unit amplitude arriving from the substrate side
    (reflected amplitude ``r'`` in the substrate, transmitted ``t'`` in the
    incident medium), matching the tangential fields at both faces gives

        r' = (V - U) / (U + V),   U = eta_0 m11 + m21
                                  V = eta_s (eta_0 m12 + m22)
        t' = m11 (1 + r') + m12 eta_s (r' - 1)

    For the identity matrix (no layers) these reduce to the bare-interface
    Fresnel coefficients taken from the substrate side, as they must.
    """
    m11, m12 = matrix[0, 0], matrix[0, 1]
    m21, m22 = matrix[1, 0], matrix[1, 1]
    u = eta_0 * m11 + m21
    v = eta_s * (eta_0 * m12 + m22)
    r_rev = (v - u) / (u + v)
    t_rev = m11 * (1.0 + r_rev) + m12 * eta_s * (r_rev - 1.0)
    reflectance = float(abs(r_rev) ** 2)
    transmittance = float((eta_0.real / eta_s.real) * abs(t_rev) ** 2)
    return _clean(reflectance), _clean(transmittance)


def interface_coefficients(eta_a: complex, eta_b: complex) -> tuple[float, float]:
    """Power R/T across a bare interface, light incident from side ``a``."""
    r = (eta_a - eta_b) / (eta_a + eta_b)
    t = (2.0 * eta_a) / (eta_a + eta_b)
    reflectance = float(abs(r) ** 2)
    transmittance = float((eta_b.real / eta_a.real) * abs(t) ** 2)
    return _clean(reflectance), _clean(transmittance)


def incoherent_cascade(
    front_reflectance: float,
    front_transmittance: float,
    front_reverse_reflectance: float,
    front_reverse_transmittance: float,
    pass_transmittance: float,
    back_reflectance: float,
    back_transmittance: float,
) -> EnergyCoefficients:
    """Closed-form incoherent sum over the substrate round trips.

    The multiply-reflected contributions form a geometric series with ratio
    ``R_f' R_b tau^2``; this function evaluates its closed-form sum (the
    module-docstring formulas) rather than truncating term by term.
    """
    tau2 = pass_transmittance * pass_transmittance
    denominator = 1.0 - front_reverse_reflectance * back_reflectance * tau2
    if denominator <= 0.0:
        # Lossless trapped-cavity limit (R_f' R_b tau^2 = 1): no light enters
        # the substrate, so the sample reflects exactly the front-stack value.
        return EnergyCoefficients(
            reflectance=front_reflectance,
            transmittance=0.0,
            absorptance=_clean(1.0 - front_reflectance),
        )
    reflectance = front_reflectance + (
        front_transmittance
        * front_reverse_transmittance
        * back_reflectance
        * tau2
        / denominator
    )
    transmittance = (
        front_transmittance * back_transmittance * pass_transmittance / denominator
    )
    absorptance = _clean(1.0 - reflectance - transmittance)
    return EnergyCoefficients(
        reflectance=reflectance, transmittance=transmittance, absorptance=absorptance
    )


def solve_sample_polarization(
    stack: Stack,
    wavelength: float,
    angle_deg: float,
    polarization: str,
    substrate_thickness: float,
    back_index: complex,
) -> SamplePolarizationResult:
    """Both calibers for one polarisation of the finite-thickness sample."""
    angle_rad = math.radians(angle_deg)
    matrix, eta_0, eta_s = stack_matrix(stack, wavelength, angle_rad, polarization)

    # Front caliber: the same matrix, the same coefficients as the
    # single-point endpoint — no parallel algorithm.
    forward = coefficients_from_matrix(matrix, eta_0, eta_s)
    front_only = EnergyCoefficients(
        reflectance=forward.reflectance,
        transmittance=forward.transmittance,
        absorptance=forward.absorptance,
    )

    # The same matrix read in the opposite direction: the front stack as
    # seen by light bouncing back up from inside the substrate.
    reverse_r, reverse_t = reverse_coefficients(matrix, eta_0, eta_s)

    # Single-pass attenuation across the finite substrate.
    sin_theta0 = math.sin(angle_rad)
    cos_s = cos_theta_in_layer(stack.substrate_index, sin_theta0, stack.incident_index)
    tau = single_pass_transmittance(
        stack.substrate_index, cos_s, substrate_thickness, wavelength
    )

    # Bare back interface: substrate -> back-side medium.
    cos_b = cos_theta_in_layer(back_index, sin_theta0, stack.incident_index)
    eta_b = effective_admittance(back_index, cos_b, polarization)
    back_r, back_t = interface_coefficients(eta_s, eta_b)

    with_back = incoherent_cascade(
        front_only.reflectance,
        front_only.transmittance,
        reverse_r,
        reverse_t,
        tau,
        back_r,
        back_t,
    )
    return SamplePolarizationResult(
        front_only=front_only,
        with_back=with_back,
        single_pass_transmittance=tau,
        back_interface_reflectance=back_r,
        back_interface_transmittance=back_t,
        front_reverse_reflectance=reverse_r,
        front_reverse_transmittance=reverse_t,
    )


def solve_sample(
    stack: Stack,
    wavelength: float,
    angle_deg: float,
    substrate_thickness: float,
    back_index: complex,
) -> SampleResult:
    """Solve both polarisations of the finite-thickness sample."""
    return SampleResult(
        wavelength=wavelength,
        angle_deg=angle_deg,
        substrate_thickness=substrate_thickness,
        back_index=back_index,
        s=solve_sample_polarization(
            stack, wavelength, angle_deg, POL_S, substrate_thickness, back_index
        ),
        p=solve_sample_polarization(
            stack, wavelength, angle_deg, POL_P, substrate_thickness, back_index
        ),
    )


def select_sample_polarization(sample: SampleResult, polarization: str) -> SamplePolarizationResult:
    """Pick one polarisation view of a :class:`SampleResult` (``s``/``p``/``avg``)."""
    if polarization == POL_S:
        return sample.s
    if polarization == POL_P:
        return sample.p
    if polarization == POL_AVG:
        return sample.averaged()
    raise ValueError(f"unknown polarisation: {polarization!r}")
