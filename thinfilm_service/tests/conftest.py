"""Shared fixtures and analytic references for the test-suite."""

from __future__ import annotations

import math

import pytest

from thinopt.matrices import Layer, Stack

#: Tolerances: matrix results are compared against analytic references.
TIGHT = 1e-12
LOOSE = 1e-9


def fresnel_rt(n0: float, n1: float, angle_deg: float, polarization: str) -> tuple[float, float]:
    """Analytic single-interface Fresnel power coefficients (textbook reference)."""
    theta0 = math.radians(angle_deg)
    sin1 = n0 * math.sin(theta0) / n1
    cos0 = math.cos(theta0)
    cos1 = math.sqrt(1.0 - sin1 * sin1)
    if polarization == "s":
        r = (n0 * cos0 - n1 * cos1) / (n0 * cos0 + n1 * cos1)
        t = 2.0 * n0 * cos0 / (n0 * cos0 + n1 * cos1)
    elif polarization == "p":
        r = (n1 * cos0 - n0 * cos1) / (n1 * cos0 + n0 * cos1)
        t = 2.0 * n0 * cos0 / (n1 * cos0 + n0 * cos1)
    else:
        raise ValueError(polarization)
    return r * r, (n1 * cos1) / (n0 * cos0) * t * t


@pytest.fixture()
def demo_design_wavelength() -> float:
    return 550.0


@pytest.fixture()
def quarter_wave_stack(demo_design_wavelength) -> Stack:
    """Air | lambda0/4 of n=1.38 | glass n=1.52 (the built-in demo coating)."""
    return Stack(
        incident_index=1.0,
        substrate_index=1.52,
        layers=(Layer(index=1.38, thickness=demo_design_wavelength / (4.0 * 1.38)),),
    )


@pytest.fixture()
def half_wave_stack(demo_design_wavelength) -> Stack:
    """Same layer material, twice the thickness: optically lambda0/2 at design."""
    return Stack(
        incident_index=1.0,
        substrate_index=1.52,
        layers=(Layer(index=1.38, thickness=demo_design_wavelength / (2.0 * 1.38)),),
    )


@pytest.fixture()
def bare_stack() -> Stack:
    """No layers at all: must reduce to the bare air/glass Fresnel interface."""
    return Stack(incident_index=1.0, substrate_index=1.52, layers=())
