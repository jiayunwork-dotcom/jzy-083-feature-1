"""Finite-substrate (incoherent back-surface) benchmarks, locked in as tests.

Each class pins one of the correctness criteria the lab agreed on:

* degenerate self-consistency: lossless substrate + back medium equal to the
  substrate makes the back interface vanish, and the with-back totals must
  collapse exactly onto the original front-only solver numbers;
* energy conservation: R + T + A = 1 with A >= 0 after the incoherent sum
  (lossless cases must satisfy R + T = 1 tightly, a nontrivial identity);
* back-surface contribution: with a good AR coating the total reflectance
  must sit well above the front-only value, by an amount matching the bare
  back interface (~4.3 % for glass in air; a bare slide totals ~8.2 %);
* series convergence: the closed-form geometric sum matches explicit
  truncation, and the multiple-bounce tail decays monotonically to zero as
  the substrate gets thicker / more absorbing.

Plus the usual guards: reverse-direction reciprocity, single-pass
attenuation, HTTP behaviour, validation, and concurrency isolation.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from thinopt import create_app
from thinopt.incoherent import (
    incoherent_cascade,
    single_pass_transmittance,
    solve_sample,
    solve_sample_polarization,
)
from thinopt.matrices import Layer, Stack
from thinopt.optics import cos_theta_in_layer
from thinopt.solver import solve_point, solve_polarization

from conftest import LOOSE, TIGHT, fresnel_rt

DESIGN_WAVELENGTH = 550.0
#: A 0.5 mm glass slide, in the same (nm) unit as the wavelength.
SLIDE_THICKNESS = 5.0e5


def _ar_stack() -> Stack:
    """Air | quarter-wave n=1.38 | glass n=1.52 (the built-in demo coating)."""
    return Stack(
        incident_index=1.0,
        substrate_index=1.52,
        layers=(Layer(index=1.38, thickness=DESIGN_WAVELENGTH / (4.0 * 1.38)),),
    )


def _bare_slide() -> Stack:
    return Stack(incident_index=1.0, substrate_index=1.52, layers=())


def _sample_request(**overrides):
    payload = {
        "incident": 1.0,
        "substrate": 1.52,
        "layers": [{"index": 1.38, "thickness": DESIGN_WAVELENGTH / (4.0 * 1.38)}],
        "wavelength": DESIGN_WAVELENGTH,
        "angle_deg": 0.0,
        "polarization": "s",
        "substrate_thickness": SLIDE_THICKNESS,
        "back_index": 1.0,
    }
    payload.update(overrides)
    return payload


class TestDegenerateLimit:
    """Back medium == substrate (lossless): the back interface vanishes and the
    with-back caliber must return the original front-only numbers exactly."""

    @pytest.mark.parametrize("wavelength", [400.0, 550.0, 700.0])
    @pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_vanishing_back_interface_recovers_front_only(
        self, wavelength, angle_deg, polarization
    ):
        stack = _ar_stack()
        result = solve_sample_polarization(
            stack, wavelength, angle_deg, polarization, SLIDE_THICKNESS, complex(1.52)
        )
        front = solve_polarization(stack, wavelength, angle_deg, polarization)
        # The back interface really vanished: no reflectance there at all.
        assert result.back_interface_reflectance == pytest.approx(0.0, abs=TIGHT)
        assert result.single_pass_transmittance == pytest.approx(1.0, abs=TIGHT)
        # Both calibers agree, and they agree with the original solver.
        assert result.with_back.reflectance == pytest.approx(front.reflectance, abs=TIGHT)
        assert result.with_back.transmittance == pytest.approx(front.transmittance, abs=TIGHT)
        assert result.front_only.reflectance == pytest.approx(front.reflectance, abs=TIGHT)
        assert result.front_only.transmittance == pytest.approx(front.transmittance, abs=TIGHT)

    def test_degenerate_bare_slide(self):
        stack = _bare_slide()
        result = solve_sample_polarization(
            stack, 550.0, 45.0, "p", SLIDE_THICKNESS, complex(1.52)
        )
        front = solve_polarization(stack, 550.0, 45.0, "p")
        assert result.with_back.reflectance == pytest.approx(front.reflectance, abs=TIGHT)
        assert result.with_back.transmittance == pytest.approx(front.transmittance, abs=TIGHT)

    def test_degenerate_with_absorbing_substrate(self):
        """Back medium == absorbing substrate: R_b = 0 still, and the only
        signature of the finite substrate is one attenuated crossing."""
        index = complex(1.52, 0.003)
        stack = Stack(
            incident_index=1.0,
            substrate_index=index,
            layers=(Layer(index=1.38, thickness=DESIGN_WAVELENGTH / (4.0 * 1.38)),),
        )
        result = solve_sample_polarization(stack, 550.0, 20.0, "s", 2.0e5, index)
        front = solve_polarization(stack, 550.0, 20.0, "s")
        tau = result.single_pass_transmittance
        assert 0.0 < tau < 1.0
        assert result.back_interface_reflectance == pytest.approx(0.0, abs=TIGHT)
        assert result.with_back.reflectance == pytest.approx(front.reflectance, abs=TIGHT)
        assert result.with_back.transmittance == pytest.approx(
            front.transmittance * tau, abs=TIGHT
        )
        assert result.with_back.absorptance > 0.0


class TestEnergyConservation:
    """After the incoherent sum: R + T + A = 1 and A >= 0, always."""

    @pytest.mark.parametrize("wavelength", [400.0, 550.0, 700.0])
    @pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0, 80.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_lossless_sample_r_plus_t_is_one(self, wavelength, angle_deg, polarization):
        """Lossless stack + lossless substrate: the cascade must conserve
        energy exactly (a nontrivial identity of the closed form)."""
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(
                Layer(index=1.38, thickness=99.6),
                Layer(index=2.1, thickness=137.0),
            ),
        )
        result = solve_sample_polarization(
            stack, wavelength, angle_deg, polarization, SLIDE_THICKNESS, complex(1.0)
        )
        total = result.with_back
        assert total.reflectance + total.transmittance == pytest.approx(1.0, abs=LOOSE)
        assert total.absorptance == pytest.approx(0.0, abs=LOOSE)
        assert 0.0 <= total.reflectance <= 1.0
        assert 0.0 <= total.transmittance <= 1.0

    @pytest.mark.parametrize("angle_deg", [0.0, 45.0, 70.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_absorbing_substrate_energy_balance(self, angle_deg, polarization):
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 0.002),
            layers=(Layer(index=1.38, thickness=99.6),),
        )
        result = solve_sample_polarization(
            stack, 550.0, angle_deg, polarization, 3.0e5, complex(1.0)
        )
        total = result.with_back
        assert (
            total.reflectance + total.transmittance + total.absorptance
            == pytest.approx(1.0, abs=LOOSE)
        )
        assert 0.0 < total.absorptance < 1.0  # the substrate genuinely absorbs
        assert 0.0 <= total.reflectance <= 1.0
        assert 0.0 <= total.transmittance <= 1.0

    def test_absorbing_films_plus_absorbing_substrate(self):
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 0.001),
            layers=(
                Layer(index=complex(1.7, 0.4), thickness=180.0),
                Layer(index=1.38, thickness=95.0),
            ),
        )
        result = solve_sample(stack, 550.0, 50.0, 2.0e5, complex(1.0))
        for pol in (result.s, result.p, result.averaged()):
            total = pol.with_back
            assert (
                total.reflectance + total.transmittance + total.absorptance
                == pytest.approx(1.0, abs=LOOSE)
            )
            assert 0.0 <= total.absorptance <= 1.0

    def test_total_internal_reflection_at_back_face(self):
        """Back face beyond the critical angle: nothing escapes, nothing is
        absorbed (lossless), so the whole sample reflects 100 %."""
        stack = Stack(incident_index=1.5, substrate_index=1.5, layers=())
        for polarization in ("s", "p"):
            result = solve_sample_polarization(
                stack, 550.0, 80.0, polarization, SLIDE_THICKNESS, complex(1.0)
            )
            assert result.back_interface_reflectance == pytest.approx(1.0, abs=TIGHT)
            assert result.with_back.reflectance == pytest.approx(1.0, abs=LOOSE)
            assert result.with_back.transmittance == pytest.approx(0.0, abs=LOOSE)
            assert result.with_back.absorptance == pytest.approx(0.0, abs=LOOSE)


class TestBackSurfaceContribution:
    """The back interface must contribute what a bare glass/air interface costs."""

    def test_ar_coated_slide_total_reflectance(self):
        result = solve_sample_polarization(
            _ar_stack(), DESIGN_WAVELENGTH, 0.0, "s", SLIDE_THICKNESS, complex(1.0)
        )
        front_r = result.front_only.reflectance
        total_r = result.with_back.reflectance
        # Front-only caliber: the familiar ~1.26 % of the quarter-wave AR.
        assert front_r == pytest.approx(0.0126, abs=1e-3)
        # The bare back interface (glass -> air) reflects ~4.26 %.
        r_back_bare = ((1.52 - 1.0) / (1.52 + 1.0)) ** 2
        assert result.back_interface_reflectance == pytest.approx(r_back_bare, abs=TIGHT)
        # Total reflectance clearly exceeds the front-only value ...
        assert total_r > front_r + 0.02
        # ... by an amount matching the bare back interface (slightly reduced
        # by the two front-stack crossings, so just under R_b itself).
        contribution = total_r - front_r
        assert 0.9 * r_back_bare < contribution < 1.05 * r_back_bare
        # Lossless: energy still balances.
        assert total_r + result.with_back.transmittance == pytest.approx(1.0, abs=LOOSE)

    def test_bare_slide_reflects_about_eight_percent(self):
        """A bare glass slide in air: two identical incoherent faces give
        R = 2R1/(1+R1) ~ 8.2 %, the textbook slide reflectance."""
        result = solve_sample_polarization(
            _bare_slide(), DESIGN_WAVELENGTH, 0.0, "s", SLIDE_THICKNESS, complex(1.0)
        )
        r1, t1 = fresnel_rt(1.0, 1.52, 0.0, "s")
        expected_r = 2.0 * r1 / (1.0 + r1)
        expected_t = (1.0 - r1) / (1.0 + r1)
        assert result.with_back.reflectance == pytest.approx(expected_r, abs=TIGHT)
        assert result.with_back.transmittance == pytest.approx(expected_t, abs=TIGHT)
        assert result.with_back.reflectance == pytest.approx(0.0817, abs=1e-3)
        # Nearly, but not exactly, twice the single-face value.
        assert result.with_back.reflectance < 2.0 * r1

    def test_back_contribution_shrinks_with_substrate_absorption(self):
        """With a lossy substrate the back contribution is attenuated by
        tau^2, so the total moves back towards the front-only value."""
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 0.001),
            layers=(Layer(index=1.38, thickness=DESIGN_WAVELENGTH / (4.0 * 1.38)),),
        )
        lossy = solve_sample_polarization(stack, 550.0, 0.0, "s", 2.0e5, complex(1.0))
        lossless = solve_sample_polarization(
            _ar_stack(), 550.0, 0.0, "s", 2.0e5, complex(1.0)
        )
        assert lossy.with_back.reflectance < lossless.with_back.reflectance
        assert lossy.with_back.reflectance > lossy.front_only.reflectance


class TestSeriesConvergence:
    """The closed-form geometric sum must match explicit truncation, and the
    multiple-bounce tail must decay monotonically to zero."""

    def test_closed_form_matches_explicit_truncation(self):
        # Synthetic coefficients with a slowly converging ratio (~0.75), so
        # the truncation genuinely needs many terms.
        coeffs = {
            "front_reflectance": 0.05,
            "front_transmittance": 0.93,
            "front_reverse_reflectance": 0.85,
            "front_reverse_transmittance": 0.10,
            "pass_transmittance": 0.99,
            "back_reflectance": 0.90,
            "back_transmittance": 0.08,
        }
        closed = incoherent_cascade(**coeffs)

        ratio = 0.85 * 0.90 * 0.99**2
        r_bounce = 0.93 * 0.10 * 0.90 * 0.99**2
        t_first = 0.93 * 0.08 * 0.99
        r_sum, t_sum = 0.05, 0.0
        term = 1.0
        for _ in range(100000):
            r_sum += r_bounce * term
            t_sum += t_first * term
            term *= ratio
            if term < 1e-18:
                break
        assert closed.reflectance == pytest.approx(r_sum, abs=TIGHT)
        assert closed.transmittance == pytest.approx(t_sum, abs=TIGHT)
        assert closed.reflectance + closed.transmittance + closed.absorptance == pytest.approx(
            1.0, abs=LOOSE
        )

    def test_closed_form_matches_truncation_end_to_end(self):
        """Same check through the public solver: rebuild the truncated series
        from the reported diagnostics and compare against the closed form."""
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 1e-5),
            layers=(Layer(index=1.38, thickness=DESIGN_WAVELENGTH / (4.0 * 1.38)),),
        )
        result = solve_sample_polarization(stack, 550.0, 0.0, "s", 1.0e5, complex(1.0))
        t_f = result.front_only.transmittance
        t_rev = result.front_reverse_transmittance
        r_rev = result.front_reverse_reflectance
        r_b = result.back_interface_reflectance
        t_b = result.back_interface_transmittance
        tau = result.single_pass_transmittance

        ratio = r_rev * r_b * tau**2
        r_bounce = t_f * t_rev * r_b * tau**2
        t_first = t_f * t_b * tau
        r_sum = result.front_only.reflectance
        t_sum = 0.0
        term = 1.0
        for _ in range(100000):
            r_sum += r_bounce * term
            t_sum += t_first * term
            term *= ratio
            if term < 1e-18:
                break
        assert result.with_back.reflectance == pytest.approx(r_sum, abs=TIGHT)
        assert result.with_back.transmittance == pytest.approx(t_sum, abs=TIGHT)

    def test_multiple_bounce_tail_decays_monotonically_to_zero(self):
        """Thicker / more absorbing substrate: the contribution of two or
        more substrate round trips shrinks monotonically towards zero."""
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 0.002),
            layers=(Layer(index=1.38, thickness=DESIGN_WAVELENGTH / (4.0 * 1.38)),),
        )
        tails = []
        transmissions = []
        for thickness in (1.0e3, 1.0e4, 5.0e4, 2.0e5, 1.0e6):
            result = solve_sample_polarization(stack, 550.0, 0.0, "s", thickness, complex(1.0))
            first_return = (
                result.front_only.transmittance
                * result.front_reverse_transmittance
                * result.back_interface_reflectance
                * result.single_pass_transmittance**2
            )
            tail = result.with_back.reflectance - result.front_only.reflectance - first_return
            tails.append(tail)
            transmissions.append(result.with_back.transmittance)
        assert all(tail >= -TIGHT for tail in tails)
        assert all(b < a for a, b in zip(tails, tails[1:]))
        assert all(b < a for a, b in zip(transmissions, transmissions[1:]))
        assert tails[-1] == pytest.approx(0.0, abs=1e-9)
        assert transmissions[-1] == pytest.approx(0.0, abs=1e-12)

    def test_thick_absorbing_substrate_recovers_front_only(self):
        """A thick, lossy substrate swallows the back channel entirely:
        R -> R_front, T -> 0, A -> 1 - R_front."""
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 0.01),
            layers=(Layer(index=1.38, thickness=DESIGN_WAVELENGTH / (4.0 * 1.38)),),
        )
        result = solve_sample_polarization(stack, 550.0, 0.0, "s", 1.0e6, complex(1.0))
        assert result.single_pass_transmittance == pytest.approx(0.0, abs=1e-60)
        assert result.with_back.reflectance == pytest.approx(
            result.front_only.reflectance, abs=TIGHT
        )
        assert result.with_back.transmittance == pytest.approx(0.0, abs=TIGHT)
        assert result.with_back.absorptance == pytest.approx(
            1.0 - result.front_only.reflectance, abs=LOOSE
        )


class TestReverseDirection:
    """The same system matrix read from the substrate side (reciprocity pins)."""

    @pytest.mark.parametrize("angle_deg", [0.0, 45.0, 70.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_lossless_stack_same_both_directions(self, angle_deg, polarization):
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(
                Layer(index=1.38, thickness=99.6),
                Layer(index=2.1, thickness=137.0),
                Layer(index=1.62, thickness=210.0),
            ),
        )
        result = solve_sample_polarization(
            stack, 550.0, angle_deg, polarization, SLIDE_THICKNESS, complex(1.0)
        )
        # Lossless reciprocal stack: R and T are direction-independent.
        assert result.front_reverse_reflectance == pytest.approx(
            result.front_only.reflectance, abs=TIGHT
        )
        assert result.front_reverse_transmittance == pytest.approx(
            result.front_only.transmittance, abs=TIGHT
        )

    @pytest.mark.parametrize("angle_deg", [0.0, 50.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_absorbing_stack_transmittance_is_reciprocal(self, angle_deg, polarization):
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(
                Layer(index=complex(1.7, 0.4), thickness=180.0),
                Layer(index=1.38, thickness=95.0),
            ),
        )
        result = solve_sample_polarization(
            stack, 550.0, angle_deg, polarization, SLIDE_THICKNESS, complex(1.0)
        )
        # Transmittance is direction-independent even with absorption ...
        assert result.front_reverse_transmittance == pytest.approx(
            result.front_only.transmittance, abs=TIGHT
        )
        # ... and the reverse direction still conserves energy.
        assert 0.0 <= result.front_reverse_reflectance
        assert result.front_reverse_reflectance + result.front_reverse_transmittance <= 1.0 + LOOSE

    @pytest.mark.parametrize("angle_deg", [0.0, 50.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_back_interface_matches_fresnel_reference(self, angle_deg, polarization):
        """The bare back interface must reproduce the textbook Fresnel
        coefficients for light leaving the substrate."""
        result = solve_sample_polarization(
            _bare_slide(), 550.0, angle_deg, polarization, SLIDE_THICKNESS, complex(1.0)
        )
        angle_in_substrate = math.degrees(
            math.asin(math.sin(math.radians(angle_deg)) / 1.52)
        )
        r_ref, t_ref = fresnel_rt(1.52, 1.0, angle_in_substrate, polarization)
        assert result.back_interface_reflectance == pytest.approx(r_ref, abs=TIGHT)
        assert result.back_interface_transmittance == pytest.approx(t_ref, abs=TIGHT)


class TestSinglePassTransmittance:
    def test_lossless_substrate_passes_everything(self):
        cos_t = cos_theta_in_layer(1.52, math.sin(math.radians(60.0)), 1.0)
        assert single_pass_transmittance(1.52, cos_t, 1.0e6, 550.0) == 1.0

    def test_beer_lambert_at_normal_incidence(self):
        k, d, wavelength = 0.001, 2.0e5, 550.0
        tau = single_pass_transmittance(complex(1.52, k), 1.0 + 0.0j, d, wavelength)
        assert tau == pytest.approx(math.exp(-4.0 * math.pi * k * d / wavelength), abs=TIGHT)

    def test_oblique_path_is_longer(self):
        index = complex(1.52, 0.001)
        d, wavelength = 2.0e5, 550.0
        cos_oblique = cos_theta_in_layer(index, math.sin(math.radians(50.0)), 1.0)
        tau_oblique = single_pass_transmittance(index, cos_oblique, d, wavelength)
        tau_normal = single_pass_transmittance(index, 1.0 + 0.0j, d, wavelength)
        # Longer oblique path -> more attenuation; and the magnitude follows
        # the path-length law exp(-4 pi k d / (lambda cos theta)).
        assert tau_oblique < tau_normal
        assert tau_oblique == pytest.approx(
            math.exp(-4.0 * math.pi * index.imag * d / (wavelength * cos_oblique.real)),
            rel=1e-3,
        )

    def test_monotonic_in_thickness_and_extinction(self):
        index = complex(1.52, 0.001)
        values = [
            single_pass_transmittance(index, 1.0 + 0.0j, d, 550.0)
            for d in (1.0e3, 1.0e4, 1.0e5, 1.0e6)
        ]
        assert all(b < a for a, b in zip(values, values[1:]))
        by_k = [
            single_pass_transmittance(complex(1.52, k), 1.0 + 0.0j, 1.0e5, 550.0)
            for k in (0.0, 1e-5, 1e-4, 1e-3)
        ]
        assert by_k[0] == 1.0
        assert all(b < a for a, b in zip(by_k, by_k[1:]))


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


class TestSampleEndpoint:
    def test_two_calibers_reported(self, client):
        body = client.post("/api/v1/sample", json=_sample_request()).get_json()
        assert body["front_only"]["reflectance"] == pytest.approx(0.0126, abs=1e-3)
        assert body["with_back"]["reflectance"] == pytest.approx(0.0541, abs=5e-4)
        assert body["with_back"]["reflectance"] > body["front_only"]["reflectance"] + 0.02
        for caliber in ("front_only", "with_back"):
            total = (
                body[caliber]["reflectance"]
                + body[caliber]["transmittance"]
                + body[caliber]["absorptance"]
            )
            assert total == pytest.approx(1.0, abs=LOOSE)
        # Diagnostics expose the back-interface physics.
        diagnostics = body["diagnostics"]
        assert diagnostics["back_interface_reflectance"] == pytest.approx(0.0426, abs=1e-3)
        assert diagnostics["single_pass_transmittance"] == pytest.approx(1.0, abs=TIGHT)
        assert set(body["components"]) == {"s", "p"}

    def test_degenerate_back_index_recovers_reflectance_endpoint(self, client):
        """Back medium == substrate: the with-back numbers must coincide with
        the original single-point endpoint, calibre for calibre."""
        request = _sample_request(back_index={"re": 1.52, "im": 0.0}, angle_deg=35.0)
        body = client.post("/api/v1/sample", json=request).get_json()
        reference = client.post("/api/v1/reflectance", json=request).get_json()
        for pol in ("s", "p"):
            sample_pol = body["components"][pol]
            reference_pol = reference["components"][pol]
            assert sample_pol["with_back"]["reflectance"] == pytest.approx(
                reference_pol["reflectance"], abs=TIGHT
            )
            assert sample_pol["with_back"]["transmittance"] == pytest.approx(
                reference_pol["transmittance"], abs=TIGHT
            )

    def test_front_only_matches_reflectance_endpoint(self, client):
        """The front-only caliber is the very same solver as the old endpoint."""
        request = _sample_request(angle_deg=40.0, back_index=1.0)
        body = client.post("/api/v1/sample", json=request).get_json()
        reference = client.post("/api/v1/reflectance", json=request).get_json()
        for pol in ("s", "p"):
            assert body["components"][pol]["front_only"]["reflectance"] == pytest.approx(
                reference["components"][pol]["reflectance"], abs=TIGHT
            )
            assert body["components"][pol]["front_only"]["transmittance"] == pytest.approx(
                reference["components"][pol]["transmittance"], abs=TIGHT
            )

    def test_avg_polarization_is_mean_of_components(self, client):
        body = client.post(
            "/api/v1/sample", json=_sample_request(angle_deg=50.0, polarization="avg")
        ).get_json()
        for caliber in ("front_only", "with_back"):
            mean_r = 0.5 * (
                body["components"]["s"][caliber]["reflectance"]
                + body["components"]["p"][caliber]["reflectance"]
            )
            assert body[caliber]["reflectance"] == pytest.approx(mean_r, abs=1e-15)

    def test_absorbing_substrate_over_http(self, client):
        request = _sample_request(
            substrate={"re": 1.52, "im": 0.005}, substrate_thickness=3.0e5
        )
        body = client.post("/api/v1/sample", json=request).get_json()
        total = body["with_back"]
        assert total["absorptance"] > 0.0
        assert (
            total["reflectance"] + total["transmittance"] + total["absorptance"]
            == pytest.approx(1.0, abs=LOOSE)
        )
        assert body["diagnostics"]["single_pass_transmittance"] < 1.0

    def test_back_index_defaults_to_air(self, client):
        request = _sample_request()
        del request["back_index"]
        response = client.post("/api/v1/sample", json=request)
        assert response.status_code == 200
        body = response.get_json()
        assert body["back_index"] == {"re": 1.0, "im": 0.0}
        assert body["diagnostics"]["back_interface_reflectance"] == pytest.approx(
            0.0426, abs=1e-3
        )


class TestSampleValidation:
    """Illegal substrate/back-side inputs are rejected pre-compute with 400."""

    @pytest.mark.parametrize(
        "override, field",
        [
            ({"substrate_thickness": 0.0}, "substrate_thickness"),
            ({"substrate_thickness": -5.0e5}, "substrate_thickness"),
            ({"substrate_thickness": "5mm"}, "substrate_thickness"),
            ({"substrate_thickness": True}, "substrate_thickness"),
            ({"back_index": {"re": 0.0, "im": 0.2}}, "back_index"),
            ({"back_index": -1.0}, "back_index"),
            ({"back_index": {"re": -1.5, "im": 0.0}}, "back_index"),
            ({"back_index": "glass"}, "back_index"),
        ],
    )
    def test_invalid_inputs_rejected(self, client, override, field):
        response = client.post("/api/v1/sample", json=_sample_request(**override))
        assert response.status_code == 400
        error = response.get_json()["error"]
        assert error["type"] == "validation_error"
        assert any(d["field"] == field for d in error["details"])

    def test_missing_substrate_thickness_rejected(self, client):
        request = _sample_request()
        del request["substrate_thickness"]
        response = client.post("/api/v1/sample", json=request)
        assert response.status_code == 400
        fields = {d["field"] for d in response.get_json()["error"]["details"]}
        assert "substrate_thickness" in fields

    def test_multiple_errors_reported_together(self, client):
        response = client.post(
            "/api/v1/sample",
            json=_sample_request(
                substrate_thickness=-1.0, back_index={"re": 0.0, "im": 0.1}, wavelength=-550.0
            ),
        )
        fields = {d["field"] for d in response.get_json()["error"]["details"]}
        assert {"substrate_thickness", "back_index", "wavelength"} <= fields

    def test_non_object_body_rejected(self, client):
        response = client.post("/api/v1/sample", json=[1, 2, 3])
        assert response.status_code == 400
        assert response.get_json()["error"]["type"] == "validation_error"


class TestSampleConcurrency:
    """Concurrent sample requests with different substrates must not interfere."""

    def test_parallel_sample_requests_isolated(self, client):
        app = create_app()

        def work(tag: int):
            thread_client = app.test_client()
            thickness = 1.0e5 + 1.0e4 * tag
            back_index = 1.0 + 0.01 * tag
            reflectances = set()
            for _ in range(5):
                body = thread_client.post(
                    "/api/v1/sample",
                    json=_sample_request(
                        substrate_thickness=thickness,
                        back_index=back_index,
                        wavelength=500.0 + tag,
                        angle_deg=float(tag),
                    ),
                ).get_json()
                assert body["substrate_thickness"] == pytest.approx(thickness)
                assert 0.0 <= body["with_back"]["reflectance"] <= 1.0
                reflectances.add(body["with_back"]["reflectance"])
            # Same input -> identical output on every retry (no cross-talk).
            assert len(reflectances) == 1
            return reflectances.pop()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, range(16)))
        # Distinct substrates genuinely produced distinct reflectances.
        assert len(set(results)) > 1
