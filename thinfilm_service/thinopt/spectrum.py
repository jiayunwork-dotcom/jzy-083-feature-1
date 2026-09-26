"""Wavelength scans: evaluate the same matrix solver over a spectral range.

Every point of the spectrum is produced by the very same
:func:`thinopt.solver.solve_polarization` call used for single-point
requests — there is no curve fitting, caching of shapes, or any other
shortcut, so a spectrum is exactly the concatenation of per-wavelength
truth values.
"""

from __future__ import annotations

from dataclasses import dataclass

from .matrices import Stack
from .solver import PolarizationResult, solve_polarization


@dataclass(frozen=True)
class SpectrumPoint:
    wavelength: float
    result: PolarizationResult


def linspace(start: float, stop: float, num: int) -> list[float]:
    """Evenly spaced wavelengths, both endpoints included."""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def scan_reflectance(
    stack: Stack,
    wavelength_start: float,
    wavelength_stop: float,
    num_points: int,
    angle_deg: float,
    polarization: str,
) -> list[SpectrumPoint]:
    """Reflectance (plus T and A for context) sampled over a wavelength range."""
    return [
        SpectrumPoint(wavelength=w, result=solve_polarization(stack, w, angle_deg, polarization))
        for w in linspace(wavelength_start, wavelength_stop, num_points)
    ]
