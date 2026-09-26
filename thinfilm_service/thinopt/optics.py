"""Core optical quantities for the characteristic-matrix method.

Conventions (fixed for the whole service):

* Time dependence ``exp(-i omega t)``; complex refractive index ``N = n + i k``
  with ``k >= 0`` for passive (absorbing) media.
* Angle of incidence ``theta0`` is measured in the incident medium, in radians.
* Propagation angle inside layer ``j`` follows Snell's law in the form

      N_j * sin(theta_j) = N_0 * sin(theta0)

  which stays valid for complex ``N_j`` (absorbing layers).
* Phase thickness of a layer (per pass):

      delta = (2 pi / wavelength) * N * d * cos(theta)

  The ``cos(theta)`` factor is essential: omitting it shifts the whole
  spectrum at oblique incidence.
* Effective optical admittance (in free-space units), polarisation dependent
  and never interchangeable:

      s-polarisation (TE):  eta = N * cos(theta)
      p-polarisation (TM):  eta = N / cos(theta)

  Both reduce to ``eta = N`` at normal incidence.
"""

from __future__ import annotations

import cmath
import math

TWO_PI = 2.0 * math.pi

#: Canonical polarisation keys accepted by the solver.
POL_S = "s"
POL_P = "p"
POL_AVG = "avg"


def cos_theta_in_layer(index: complex, sin_theta0: float, index_incident: float) -> complex:
    """Cosine of the propagation angle inside a layer, from Snell's law.

    ``sin(theta_j) = (N_0 / N_j) * sin(theta0)`` and
    ``cos(theta_j) = sqrt(1 - sin^2(theta_j))``.

    The square-root branch is chosen so that the wave in a passive medium
    propagates/decays towards +z: prefer ``Re(cos) >= 0`` and require
    ``Im(N * cos) >= 0`` (amplitude decays, never grows, in a passive layer).
    """
    s = (index_incident / index) * sin_theta0
    c = cmath.sqrt(1.0 - s * s)
    # Pick the physical branch for passive media.
    if c.real < 0.0:
        c = -c
    if (index * c).imag < 0.0:
        c = -c
    if c.real < 0.0:  # both criteria cannot be met (gain medium); keep decay branch
        c = -c
    return c


def effective_admittance(index: complex, cos_theta: complex, polarization: str) -> complex:
    """Effective optical admittance of a medium for one polarisation.

    s-polarisation: ``eta = N * cos(theta)``
    p-polarisation: ``eta = N / cos(theta)``

    The two expressions are independent by definition; mixing them up is a
    physics error, so the caller must pass the polarisation explicitly.
    """
    if polarization == POL_S:
        return index * cos_theta
    if polarization == POL_P:
        return index / cos_theta
    raise ValueError(f"unknown polarisation for admittance: {polarization!r}")


def phase_thickness(index: complex, thickness: float, cos_theta: complex, wavelength: float) -> complex:
    """Phase thickness ``delta = (2 pi / lambda) * N * d * cos(theta)``.

    ``thickness`` is the *physical* thickness, in the same unit as
    ``wavelength``.  The cos(theta) factor accounts for the oblique path
    through the layer; dropping it shifts spectral features at non-normal
    incidence.
    """
    return TWO_PI * index * thickness * cos_theta / wavelength
