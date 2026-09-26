"""Characteristic (transfer) matrices of a multilayer stack.

Convention: time dependence ``exp(-i omega t)``, complex index
``N = n + i k`` with ``k >= 0`` for passive (absorbing) media.  With this
convention each layer ``j`` contributes the 2x2 complex matrix

    M_j = [[cos(delta_j),    -i sin(delta_j) / eta_j],
           [-i eta_j sin(delta_j),        cos(delta_j)  ]]

built from its phase thickness ``delta_j`` and its polarisation-dependent
effective admittance ``eta_j`` (see :mod:`thinopt.optics`).  (The ``-i``
sign is the one consistent with ``exp(-i omega t)`` and ``N = n + i k``;
textbooks using ``exp(+i omega t)`` write ``+i`` together with ``n - i k``.
Mixing the two conventions turns absorption into gain.)  The matrices are
multiplied in incidence order, layer 1 (next to the incident medium) first:

    M = M_1 * M_2 * ... * M_k

so that ``[B; C] = M @ [1; eta_substrate]`` gives the normalised tangential
field amplitudes at the entrance face.  All arithmetic is complex, so
absorbing layers (complex index) need no special casing.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from .optics import cos_theta_in_layer, effective_admittance, phase_thickness


@dataclass(frozen=True)
class Layer:
    """One homogeneous layer: complex refractive index and physical thickness.

    ``thickness`` uses the same length unit as the wavelength passed to the
    solver (typically nm).
    """

    index: complex
    thickness: float


@dataclass(frozen=True)
class Stack:
    """A complete multilayer system.

    ``incident_index`` must be real (lossless incident medium) so that the
    incident energy flux is well defined; layers and substrate may be complex.
    """

    incident_index: float
    substrate_index: complex
    layers: tuple[Layer, ...]


def layer_matrix(delta: complex, admittance: complex) -> np.ndarray:
    """Characteristic matrix of a single layer from its phase thickness and admittance."""
    cos_d = cmath.cos(delta)
    sin_d = cmath.sin(delta)
    return np.array(
        [
            [cos_d, -1j * sin_d / admittance],
            [-1j * admittance * sin_d, cos_d],
        ],
        dtype=complex,
    )


def boundary_admittances(
    index: complex,
    polarization: str,
    sin_theta0: float,
    index_incident: float,
) -> complex:
    """Effective admittance of a semi-infinite bounding medium (incident side or substrate)."""
    cos_t = cos_theta_in_layer(index, sin_theta0, index_incident)
    return effective_admittance(index, cos_t, polarization)


def stack_matrix(
    stack: Stack,
    wavelength: float,
    angle_rad: float,
    polarization: str,
) -> tuple[np.ndarray, complex, complex]:
    """Multiply the layer matrices and return the system matrix plus boundary admittances.

    Returns ``(M, eta_incident, eta_substrate)`` where ``M`` is the 2x2
    product of all layer matrices (the 2x2 identity when there are no
    layers, which makes the result reduce exactly to the single-interface
    Fresnel coefficients between incident medium and substrate).
    """
    sin_theta0 = math.sin(angle_rad)
    matrix = np.identity(2, dtype=complex)
    for layer in stack.layers:
        cos_t = cos_theta_in_layer(layer.index, sin_theta0, stack.incident_index)
        eta = effective_admittance(layer.index, cos_t, polarization)
        delta = phase_thickness(layer.index, layer.thickness, cos_t, wavelength)
        matrix = matrix @ layer_matrix(delta, eta)
    eta_incident = boundary_admittances(
        stack.incident_index, polarization, sin_theta0, stack.incident_index
    )
    eta_substrate = boundary_admittances(
        stack.substrate_index, polarization, sin_theta0, stack.incident_index
    )
    return matrix, eta_incident, eta_substrate
