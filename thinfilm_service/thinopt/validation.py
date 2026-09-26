"""Request validation: turn raw JSON payloads into checked domain objects.

Every physical constraint is enforced here, *before* any matrix is built:

* layer thickness must be >= 0 (negative thickness is rejected);
* refractive indices must have a strictly positive real part;
* the incident medium must be lossless (real index), otherwise the incident
  energy flux — and hence R/T/A — is not well defined;
* wavelength must be > 0;
* angle of incidence must be in [0, 90) degrees;
* polarisation must be one of ``s`` / ``p`` / ``avg``.

Violations raise :class:`ValidationError` carrying structured per-field
details, which the HTTP layer serialises into the JSON error body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .matrices import Layer, Stack
from .optics import POL_AVG, POL_P, POL_S

POLARIZATIONS = (POL_S, POL_P, POL_AVG)
DEFAULT_POLARIZATION = POL_S
DEFAULT_SPECTRUM_POINTS = 201
MAX_SPECTRUM_POINTS = 20001


@dataclass(frozen=True)
class FieldError:
    field: str
    message: str


class ValidationError(Exception):
    """One or more request fields failed validation."""

    def __init__(self, errors: list[FieldError]):
        self.errors = errors
        super().__init__("; ".join(f"{e.field}: {e.message}" for e in errors))


def _is_finite_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_index(value: Any, field: str, errors: list[FieldError]) -> complex:
    """Parse a refractive index given as a real number, ``{"re","im"}`` or ``[re, im]``.

    The real part must be strictly positive; the imaginary part (extinction
    coefficient, >= 0 for passive media) defaults to 0.
    """
    re_part: Any = None
    im_part: Any = 0.0
    if _is_finite_real(value):
        re_part = value
    elif isinstance(value, dict):
        re_part = value.get("re")
        im_part = value.get("im", 0.0)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        re_part, im_part = value
    else:
        errors.append(FieldError(field, "refractive index must be a number, {'re','im'} or [re, im]"))
        return complex(1.0, 0.0)

    if not _is_finite_real(re_part) or not _is_finite_real(im_part):
        errors.append(FieldError(field, "refractive index parts must be finite numbers"))
        return complex(1.0, 0.0)
    if re_part <= 0.0:
        errors.append(FieldError(field, "refractive index real part must be > 0"))
    return complex(float(re_part), float(im_part))


def parse_layers(value: Any, errors: list[FieldError]) -> tuple[Layer, ...]:
    """Parse the layer list; each entry needs ``index`` and ``thickness``."""
    if value is None:
        return ()
    if not isinstance(value, list):
        errors.append(FieldError("layers", "must be a list of {'index', 'thickness'} objects"))
        return ()

    layers: list[Layer] = []
    for i, item in enumerate(value):
        field = f"layers[{i}]"
        if not isinstance(item, dict):
            errors.append(FieldError(field, "layer must be an object with 'index' and 'thickness'"))
            continue
        index = parse_index(item.get("index"), f"{field}.index", errors)
        thickness = item.get("thickness")
        if not _is_finite_real(thickness):
            errors.append(FieldError(f"{field}.thickness", "layer thickness must be a finite number"))
            continue
        if thickness < 0.0:
            errors.append(FieldError(f"{field}.thickness", "layer thickness must be >= 0"))
            continue
        layers.append(Layer(index=index, thickness=float(thickness)))
    return tuple(layers)


def parse_stack(payload: dict[str, Any], errors: list[FieldError]) -> Stack:
    """Parse incident medium, substrate and layers into a :class:`Stack`."""
    incident = parse_index(payload.get("incident"), "incident", errors)
    if incident.imag != 0.0:
        errors.append(FieldError("incident", "incident medium must be lossless (real refractive index)"))
    substrate = parse_index(payload.get("substrate"), "substrate", errors)
    layers = parse_layers(payload.get("layers"), errors)
    return Stack(incident_index=incident.real, substrate_index=substrate, layers=layers)


def parse_wavelength(value: Any, field: str, errors: list[FieldError]) -> float:
    """Wavelength must be a finite number > 0 (same unit as layer thicknesses)."""
    if not _is_finite_real(value):
        errors.append(FieldError(field, "wavelength must be a finite number"))
        return 1.0
    if value <= 0.0:
        errors.append(FieldError(field, "wavelength must be > 0"))
    return float(value)


def parse_angle(value: Any, errors: list[FieldError]) -> float:
    """Angle of incidence in degrees, must satisfy 0 <= angle < 90."""
    if not _is_finite_real(value):
        errors.append(FieldError("angle_deg", "angle of incidence must be a finite number"))
        return 0.0
    if not 0.0 <= value < 90.0:
        errors.append(FieldError("angle_deg", "angle of incidence must be in [0, 90) degrees"))
    return float(value)


def parse_polarization(value: Any, errors: list[FieldError]) -> str:
    """Polarisation selector: 's', 'p' or 'avg' (unpolarised mean)."""
    if value is None:
        return DEFAULT_POLARIZATION
    if isinstance(value, str) and value.lower() in POLARIZATIONS:
        return value.lower()
    errors.append(FieldError("polarization", f"must be one of {list(POLARIZATIONS)}"))
    return DEFAULT_POLARIZATION


def parse_num_points(value: Any, errors: list[FieldError]) -> int:
    """Number of spectrum samples; integer in [2, MAX_SPECTRUM_POINTS]."""
    if value is None:
        return DEFAULT_SPECTRUM_POINTS
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(FieldError("points", "number of points must be an integer"))
        return DEFAULT_SPECTRUM_POINTS
    if not 2 <= value <= MAX_SPECTRUM_POINTS:
        errors.append(FieldError("points", f"number of points must be in [2, {MAX_SPECTRUM_POINTS}]"))
    return value


@dataclass(frozen=True)
class PointRequest:
    stack: Stack
    wavelength: float
    angle_deg: float
    polarization: str


@dataclass(frozen=True)
class SpectrumRequest:
    stack: Stack
    wavelength_start: float
    wavelength_stop: float
    num_points: int
    angle_deg: float
    polarization: str


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError([FieldError("body", "request body must be a JSON object")])
    return payload


def parse_point_request(payload: Any) -> PointRequest:
    """Validate a single-point calculation request, or raise ValidationError."""
    body = _require_object(payload)
    errors: list[FieldError] = []
    stack = parse_stack(body, errors)
    wavelength = parse_wavelength(body.get("wavelength"), "wavelength", errors)
    angle = parse_angle(body.get("angle_deg", 0.0), errors)
    polarization = parse_polarization(body.get("polarization"), errors)
    if errors:
        raise ValidationError(errors)
    return PointRequest(stack=stack, wavelength=wavelength, angle_deg=angle, polarization=polarization)


def parse_spectrum_request(payload: Any) -> SpectrumRequest:
    """Validate a spectral-scan request, or raise ValidationError."""
    body = _require_object(payload)
    errors: list[FieldError] = []
    stack = parse_stack(body, errors)
    start = parse_wavelength(body.get("wavelength_start"), "wavelength_start", errors)
    stop = parse_wavelength(body.get("wavelength_stop"), "wavelength_stop", errors)
    if not any(e.field.startswith("wavelength_") for e in errors) and stop <= start:
        errors.append(FieldError("wavelength_stop", "must be greater than wavelength_start"))
    num_points = parse_num_points(body.get("points"), errors)
    angle = parse_angle(body.get("angle_deg", 0.0), errors)
    polarization = parse_polarization(body.get("polarization"), errors)
    if errors:
        raise ValidationError(errors)
    return SpectrumRequest(
        stack=stack,
        wavelength_start=start,
        wavelength_stop=stop,
        num_points=num_points,
        angle_deg=angle,
        polarization=polarization,
    )
