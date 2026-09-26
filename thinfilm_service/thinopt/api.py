"""HTTP interface: JSON in, JSON out, no HTML anywhere.

Endpoints
---------
GET  /health                     liveness probe
POST /api/v1/reflectance         single point: stack + wavelength + angle -> R/T/A
POST /api/v1/reflectance/slab    whole sample: finite substrate + back face -> R/T/A
POST /api/v1/spectrum            wavelength scan: stack + range -> sampled spectrum
GET  /api/v1/example             built-in quarter-wave AR demo, with expected values

The layer is deliberately thin: parsing/validation lives in
:mod:`thinopt.validation`, physics in :mod:`thinopt.solver` /
:mod:`thinopt.spectrum` / :mod:`thinopt.substrate`.  Handlers hold no
mutable state, so concurrent requests (each with its own stack and matrix
intermediates) never interfere.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request

from . import presets
from .matrices import Stack
from .solver import PointResult, PolarizationResult, select_polarization, solve_point
from .spectrum import scan_reflectance
from .substrate import (
    SlabPointResult,
    SlabPolarizationResult,
    select_slab_polarization,
    solve_slab,
)
from .validation import (
    SlabRequest,
    ValidationError,
    parse_point_request,
    parse_slab_request,
    parse_spectrum_request,
)

JSON = "application/json"


def _complex_json(value: complex) -> dict[str, float]:
    return {"re": value.real, "im": value.imag}


def _coefficients_json(result: PolarizationResult) -> dict[str, Any]:
    return {
        "reflectance": result.reflectance,
        "transmittance": result.transmittance,
        "absorptance": result.absorptance,
        "amplitude_r": _complex_json(result.r),
        "amplitude_t": _complex_json(result.t),
    }


def _stack_json(stack: Stack) -> dict[str, Any]:
    return {
        "incident": stack.incident_index,
        "substrate": {"re": stack.substrate_index.real, "im": stack.substrate_index.imag},
        "layers": [
            {"index": {"re": layer.index.real, "im": layer.index.imag}, "thickness": layer.thickness}
            for layer in stack.layers
        ],
    }


def _point_response(point: PointResult, polarization: str) -> dict[str, Any]:
    return {
        "wavelength": point.wavelength,
        "angle_deg": point.angle_deg,
        "polarization": polarization,
        **_coefficients_json(select_polarization(point, polarization)),
        "components": {
            "s": _coefficients_json(point.s),
            "p": _coefficients_json(point.p),
        },
    }


def _rta_json(reflectance: float, transmittance: float, absorptance: float) -> dict[str, float]:
    return {
        "reflectance": reflectance,
        "transmittance": transmittance,
        "absorptance": absorptance,
    }


def _slab_coefficients_json(result: SlabPolarizationResult) -> dict[str, Any]:
    """Both accounting scopes side by side: front-only coherent vs whole sample."""
    return {
        **_rta_json(result.reflectance, result.transmittance, result.absorptance),
        "front_only": _rta_json(
            result.front.reflectance, result.front.transmittance, result.front.absorptance
        ),
        "back_contribution": _rta_json(
            result.reflectance - result.front.reflectance,
            result.transmittance - result.front.transmittance,
            result.absorptance - result.front.absorptance,
        ),
        "back_interface_reflectance": result.back.reflectance,
        "round_trip_factor": result.round_trip_factor,
    }


def _slab_response(point: SlabPointResult, req: SlabRequest) -> dict[str, Any]:
    return {
        "wavelength": point.wavelength,
        "angle_deg": point.angle_deg,
        "polarization": req.polarization,
        "substrate": {
            "thickness": req.substrate.thickness,
            "back_index": _complex_json(req.substrate.back_index),
            # alpha is polarisation-independent; the s view carries it.
            "single_pass_transmittance": point.s.single_pass_transmittance,
        },
        **_slab_coefficients_json(select_slab_polarization(point, req.polarization)),
        "components": {
            "s": _slab_coefficients_json(point.s),
            "p": _slab_coefficients_json(point.p),
        },
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.errorhandler(ValidationError)
    def _validation_error(exc: ValidationError):
        body = {
            "error": {
                "type": "validation_error",
                "message": "request failed validation",
                "details": [{"field": e.field, "message": e.message} for e in exc.errors],
            }
        }
        return jsonify(body), 400

    @app.errorhandler(404)
    def _not_found(_exc):
        return jsonify({"error": {"type": "not_found", "message": "unknown endpoint"}}), 404

    @app.errorhandler(405)
    def _method_not_allowed(_exc):
        return jsonify({"error": {"type": "method_not_allowed", "message": "method not allowed"}}), 405

    @app.errorhandler(500)
    def _internal_error(_exc):
        return jsonify({"error": {"type": "internal_error", "message": "unexpected server error"}}), 500

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/v1/reflectance")
    def reflectance():
        # get_json(silent=True) yields None for missing/malformed JSON, which
        # parse_point_request rejects with a structured validation error.
        req = parse_point_request(request.get_json(silent=True))
        point = solve_point(req.stack, req.wavelength, req.angle_deg)
        return jsonify(_point_response(point, req.polarization))

    @app.post("/api/v1/reflectance/slab")
    def reflectance_slab():
        # Whole sample: coherent front stack + incoherent finite substrate.
        req = parse_slab_request(request.get_json(silent=True))
        point = solve_slab(req.stack, req.substrate, req.wavelength, req.angle_deg)
        return jsonify(_slab_response(point, req))

    @app.post("/api/v1/spectrum")
    def spectrum():
        req = parse_spectrum_request(request.get_json(silent=True))
        points = scan_reflectance(
            req.stack,
            req.wavelength_start,
            req.wavelength_stop,
            req.num_points,
            req.angle_deg,
            req.polarization,
        )
        return jsonify(
            {
                "angle_deg": req.angle_deg,
                "polarization": req.polarization,
                "points": [
                    {
                        "wavelength": p.wavelength,
                        "reflectance": p.result.reflectance,
                        "transmittance": p.result.transmittance,
                        "absorptance": p.result.absorptance,
                    }
                    for p in points
                ],
            }
        )

    @app.get("/api/v1/example")
    def example():
        """Quarter-wave AR demo, evaluated live so the numbers always match the solver."""
        coated = presets.demo_stack()
        bare = presets.bare_stack()
        wavelength = presets.DESIGN_WAVELENGTH
        coated_r = select_polarization(solve_point(coated, wavelength, 0.0), "s").reflectance
        bare_r = select_polarization(solve_point(bare, wavelength, 0.0), "s").reflectance
        return jsonify(
            {
                "description": "single quarter-wave anti-reflection layer on glass",
                "design_wavelength": wavelength,
                "coated": {
                    "request": {
                        "incident": presets.INCIDENT_INDEX,
                        "substrate": presets.SUBSTRATE_INDEX,
                        "layers": [
                            {"index": presets.LAYER_INDEX, "thickness": presets.QUARTER_WAVE_THICKNESS}
                        ],
                        "wavelength": wavelength,
                        "angle_deg": 0.0,
                        "polarization": "s",
                    },
                    "reflectance": coated_r,
                },
                "bare": {
                    "request": {
                        "incident": presets.INCIDENT_INDEX,
                        "substrate": presets.SUBSTRATE_INDEX,
                        "layers": [],
                        "wavelength": wavelength,
                        "angle_deg": 0.0,
                        "polarization": "s",
                    },
                    "reflectance": bare_r,
                },
                "expected": {
                    "relation": "coated reflectance << bare reflectance at the design wavelength",
                    "holds": bool(coated_r < bare_r),
                },
            }
        )

    return app
