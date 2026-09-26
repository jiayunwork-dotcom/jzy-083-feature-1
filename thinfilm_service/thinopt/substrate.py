"""Finite-thickness substrate: the incoherent back-face channel.

A real sample is a slab of glass, not a semi-infinite half-space: behind
the coating there is a second, bare substrate/back-medium interface that
reflects part of the light back towards the detector.  When the front
coating pushes the front-face reflectance below a percent, that bare back
interface (a few percent for glass/air) dominates the measured signal, so
a front-only calculation never matches the bench.

Physics split (the crucial modelling decision):

* The front stack (incident medium -> layers -> substrate) is thin and
  stays fully coherent.  It is solved by the *same* characteristic-matrix
  solver as the single-point endpoint — once from the incident side and
  once, with the layer order reversed, from inside the substrate.  No
  second algorithm lives here.
* The substrate itself is a finite slab, thick compared with the coherence
  length of the light, so the phase between successive passes is washed
  out.  Multiple reflections between its two faces are summed as
  *intensities* (power), never as amplitudes.

With the coherent ingredients

    R_f, T_f    front stack, incident side
    R'_f, T'_f  front stack, seen from inside the substrate
    R_b, T_b    bare back interface (substrate -> back medium)
    alpha       single-pass intensity transmittance of the substrate,
                exp(-2 Im(delta_s)) from its phase thickness delta_s

each round trip inside the substrate multiplies the trapped intensity by
``q = R'_f * R_b * alpha^2``, so the bounce series is geometric and is
summed in closed form (not truncated term by term):

    R_total = R_f + T_f * T'_f * R_b * alpha^2 / (1 - q)
    T_total = T_f * T_b * alpha / (1 - q)
    A_total = 1 - R_total - T_total

Sanity anchors locked by the test-suite: a lossless substrate whose back
medium matches its own index makes the back interface vanish
(R_b = 0, alpha = 1) and both totals reduce *exactly* to the front-only
coherent values; and R_total + T_total + A_total = 1 with A_total >= 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .matrices import Stack
from .optics import POL_AVG, POL_P, POL_S, cos_theta_in_layer, phase_thickness
from .solver import PolarizationResult, solve_polarization, solve_polarization_sine

#: Residuals below this magnitude are floating-point noise, not physics
#: (same convention as ``solver._clean``).
_NOISE_FLOOR = 1e-12


@dataclass(frozen=True)
class FiniteSubstrate:
    """The substrate as a finite slab.

    ``thickness`` is the physical thickness, in the same unit as the
    wavelength (0.5 mm = 5e5 nm); ``back_index`` is the refractive index of
    the medium behind the substrate (typically air, 1.0).  The slab material
    itself is ``Stack.substrate_index``.
    """

    thickness: float
    back_index: complex


@dataclass(frozen=True)
class SlabPolarizationResult:
    """Whole-sample energy coefficients for one polarisation.

    ``front`` / ``reverse`` / ``back`` are the coherent ingredients (front
    stack from each side, bare back interface); ``reflectance`` /
    ``transmittance`` / ``absorptance`` are the measurable totals of the
    whole sample, front stack plus the incoherent back-face channel.
    """

    front: PolarizationResult
    reverse: PolarizationResult
    back: PolarizationResult
    single_pass_transmittance: float
    reflectance: float
    transmittance: float
    absorptance: float

    @property
    def round_trip_factor(self) -> float:
        """Intensity multiplier per substrate round trip, q = R'_f * R_b * alpha^2."""
        return (
            self.reverse.reflectance
            * self.back.reflectance
            * self.single_pass_transmittance**2
        )


@dataclass(frozen=True)
class SlabPointResult:
    """Whole-sample result at one (wavelength, angle); ``s``/``p`` per polarisation."""

    wavelength: float
    angle_deg: float
    s: SlabPolarizationResult
    p: SlabPolarizationResult

    def averaged(self) -> SlabPolarizationResult:
        """Unpolarised light: plain mean of the s and p energy coefficients."""
        s, p = self.s, self.p
        return SlabPolarizationResult(
            front=_mean_polarization(s.front, p.front),
            reverse=_mean_polarization(s.reverse, p.reverse),
            back=_mean_polarization(s.back, p.back),
            single_pass_transmittance=0.5
            * (s.single_pass_transmittance + p.single_pass_transmittance),
            reflectance=0.5 * (s.reflectance + p.reflectance),
            transmittance=0.5 * (s.transmittance + p.transmittance),
            absorptance=0.5 * (s.absorptance + p.absorptance),
        )


def _mean_polarization(s: PolarizationResult, p: PolarizationResult) -> PolarizationResult:
    """Unpolarised mean of two coherent results (mirrors PointResult.averaged)."""
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


def single_pass_transmittance(
    index: complex,
    thickness: float,
    cos_theta: complex,
    wavelength: float,
) -> float:
    """Intensity transmittance of one pass through the slab: exp(-2 Im(delta)).

    ``delta`` is the phase thickness of the slab; its imaginary part is the
    Beer-Lambert attenuation, >= 0 for passive media, so the result lies in
    [0, 1].  A lossless slab gives exactly 1.
    """
    delta = phase_thickness(index, thickness, cos_theta, wavelength)
    return math.exp(-2.0 * delta.imag)


def incoherent_cascade(
    front: PolarizationResult,
    reverse: PolarizationResult,
    back: PolarizationResult,
    alpha: float,
) -> tuple[float, float, float]:
    """Closed-form geometric sum of the substrate bounce series -> (R, T, A).

    Light either reflects off the front stack directly (R_f) or enters the
    substrate and rattles between its two faces.  Each round trip
    attenuates the trapped intensity by ``q = R'_f * R_b * alpha^2 < 1``,
    so the two intensity series

        R_f + T_f T'_f R_b alpha^2 * (1 + q + q^2 + ...)
            T_f T_b alpha         * (1 + q + q^2 + ...)

    collapse to their closed forms.  The series is summed analytically —
    truncating it term by term would be both slower and less exact.
    """
    q = reverse.reflectance * back.reflectance * alpha * alpha
    if q >= 1.0:
        # Perfectly trapped light (lossless slab, both faces totally
        # reflecting, no front transmission): nothing crosses the slab and
        # nothing is absorbed inside it, so the sample is just the front.
        return front.reflectance, 0.0, front.absorptance
    denom = 1.0 - q
    reflectance = front.reflectance + (
        front.transmittance * reverse.transmittance * back.reflectance * alpha * alpha / denom
    )
    transmittance = front.transmittance * back.transmittance * alpha / denom
    absorptance = 1.0 - reflectance - transmittance
    return _clean(reflectance), _clean(transmittance), _clean(absorptance)


def solve_slab_polarization(
    stack: Stack,
    substrate: FiniteSubstrate,
    wavelength: float,
    angle_deg: float,
    polarization: str,
) -> SlabPolarizationResult:
    """Whole-sample R/T/A for one polarisation: coherent front + incoherent slab."""
    sin_theta0 = math.sin(math.radians(angle_deg))

    # Coherent front stack from the incident side — the established solver.
    front = solve_polarization(stack, wavelength, angle_deg, polarization)

    # Single-pass attenuation across the substrate slab.
    cos_substrate = cos_theta_in_layer(stack.substrate_index, sin_theta0, stack.incident_index)
    alpha = single_pass_transmittance(
        stack.substrate_index, substrate.thickness, cos_substrate, wavelength
    )

    # The same matrix solver, seen from inside the substrate.  Only the
    # conserved tangential wavevector N*sin(theta) propagates between the
    # two viewpoints, so the propagation sine in the (possibly absorbing)
    # substrate is complex — the sine entry point handles that as-is.
    sin_theta_substrate = stack.incident_index * sin_theta0 / stack.substrate_index
    reversed_stack = Stack(
        incident_index=stack.substrate_index,
        substrate_index=stack.incident_index,
        layers=tuple(reversed(stack.layers)),
    )
    reverse = solve_polarization_sine(
        reversed_stack, wavelength, sin_theta_substrate, polarization
    )
    # Bare back interface: zero layers reduces the solver to the exact
    # single-interface Fresnel coefficients, substrate -> back medium.
    back_interface = Stack(
        incident_index=stack.substrate_index,
        substrate_index=substrate.back_index,
        layers=(),
    )
    back = solve_polarization_sine(
        back_interface, wavelength, sin_theta_substrate, polarization
    )

    reflectance, transmittance, absorptance = incoherent_cascade(front, reverse, back, alpha)
    return SlabPolarizationResult(
        front=front,
        reverse=reverse,
        back=back,
        single_pass_transmittance=alpha,
        reflectance=reflectance,
        transmittance=transmittance,
        absorptance=absorptance,
    )


def solve_slab(
    stack: Stack,
    substrate: FiniteSubstrate,
    wavelength: float,
    angle_deg: float,
) -> SlabPointResult:
    """Solve both polarisations for the whole sample at one wavelength and angle."""
    return SlabPointResult(
        wavelength=wavelength,
        angle_deg=angle_deg,
        s=solve_slab_polarization(stack, substrate, wavelength, angle_deg, POL_S),
        p=solve_slab_polarization(stack, substrate, wavelength, angle_deg, POL_P),
    )


def select_slab_polarization(point: SlabPointResult, polarization: str) -> SlabPolarizationResult:
    """Pick one polarisation view of a :class:`SlabPointResult` (``s``/``p``/``avg``)."""
    if polarization == POL_S:
        return point.s
    if polarization == POL_P:
        return point.p
    if polarization == POL_AVG:
        return point.averaged()
    raise ValueError(f"unknown polarisation: {polarization!r}")
