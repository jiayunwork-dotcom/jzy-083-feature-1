"""Physics benchmarks, locked in as tests.

Each test pins one of the correctness criteria the lab agreed on:

* lossless stacks conserve energy: R + T = 1 at any wavelength/angle;
* absorbing stacks: R + A + T = 1 with A >= 0;
* a layer-free system reduces exactly to the single-interface Fresnel result;
* a quarter-wave AR layer beats the bare substrate at the design wavelength;
* a half-wave layer is optically absent at the design wavelength;
* reflectance rises towards grazing incidence;
* the oblique-incidence dip shift proves the cos(theta) factor in the phase.
"""

from __future__ import annotations

import cmath
import math

import pytest

from thinopt.matrices import Layer, Stack
from thinopt.solver import solve_point, solve_polarization
from thinopt.spectrum import scan_reflectance

from conftest import LOOSE, TIGHT, fresnel_rt


class TestEnergyConservation:
    """Lossless stacks: R + T == 1 for arbitrary wavelength and angle."""

    @pytest.mark.parametrize("wavelength", [400.0, 550.0, 700.0, 1234.5])
    @pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0, 85.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_lossless_stack_energy_balance(self, wavelength, angle_deg, polarization):
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(
                Layer(index=1.38, thickness=99.6),
                Layer(index=2.1, thickness=137.0),
                Layer(index=1.62, thickness=210.0),
            ),
        )
        result = solve_polarization(stack, wavelength, angle_deg, polarization)
        assert result.reflectance + result.transmittance == pytest.approx(1.0, abs=LOOSE)
        assert result.absorptance == pytest.approx(0.0, abs=LOOSE)

    def test_unpolarised_energy_balance(self, quarter_wave_stack):
        point = solve_point(quarter_wave_stack, 550.0, 45.0)
        avg = point.averaged()
        assert avg.reflectance + avg.transmittance == pytest.approx(1.0, abs=LOOSE)


class TestAbsorption:
    """Absorbing stacks: R + A + T == 1, and A must never be negative."""

    @pytest.mark.parametrize("wavelength", [450.0, 550.0, 650.0])
    @pytest.mark.parametrize("angle_deg", [0.0, 45.0, 70.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_absorbing_layer_energy_balance(self, wavelength, angle_deg, polarization):
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(
                Layer(index=complex(1.7, 0.4), thickness=180.0),
                Layer(index=1.38, thickness=95.0),
            ),
        )
        result = solve_polarization(stack, wavelength, angle_deg, polarization)
        assert result.reflectance + result.transmittance + result.absorptance == pytest.approx(
            1.0, abs=LOOSE
        )
        assert result.absorptance >= 0.0
        assert result.absorptance > 1e-6  # this stack genuinely absorbs

    def test_metal_like_layer(self):
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(Layer(index=complex(0.18, 2.4), thickness=25.0),),
        )
        result = solve_polarization(stack, 550.0, 20.0, "s")
        assert result.reflectance + result.transmittance + result.absorptance == pytest.approx(
            1.0, abs=LOOSE
        )
        assert 0.0 <= result.absorptance <= 1.0


class TestAbsorbingFilmAgainstAiryFormula:
    """Cross-check the matrix solver against the independent Airy-sum formula.

    For a single film, summing the multiply-reflected amplitudes gives

        r = (r01 + r12 exp(2 i delta)) / (1 + r01 r12 exp(2 i delta))

    with interface Fresnel coefficients r01, r12 and the round-trip phase
    2 delta.  This is derived independently of the characteristic matrix,
    so agreement locks both the sign conventions and the phase definition.
    """

    @staticmethod
    def _airy_r(n0, n1, ns, d, wavelength, angle_deg, polarization):
        theta0 = math.radians(angle_deg)
        sin1 = n0 * math.sin(theta0) / n1
        cos0 = math.cos(theta0)
        cos1 = cmath.sqrt(1.0 - sin1 * sin1)
        if (n1 * cos1).imag < 0.0:
            cos1 = -cos1
        sins = n0 * math.sin(theta0) / ns
        coss = cmath.sqrt(1.0 - sins * sins)
        if polarization == "s":
            e0, e1, es = n0 * cos0, n1 * cos1, ns * coss
        else:
            e0, e1, es = n0 / cos0, n1 / cos1, ns / coss
        r01 = (e0 - e1) / (e0 + e1)
        r12 = (e1 - es) / (e1 + es)
        delta = 2.0 * math.pi * n1 * d * cos1 / wavelength
        phase = cmath.exp(2j * delta)
        return (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)

    @pytest.mark.parametrize("angle_deg", [0.0, 40.0, 65.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_single_absorbing_film_matches_airy(self, angle_deg, polarization):
        n1 = complex(1.7, 0.4)
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(Layer(index=n1, thickness=180.0),),
        )
        result = solve_polarization(stack, 550.0, angle_deg, polarization)
        r_ref = self._airy_r(1.0, n1, 1.52, 180.0, 550.0, angle_deg, polarization)
        assert result.reflectance == pytest.approx(abs(r_ref) ** 2, abs=LOOSE)


class TestBareInterfaceLimit:
    """No layers => exactly the single-interface Fresnel coefficients."""

    @pytest.mark.parametrize("angle_deg", [0.0, 15.0, 45.0, 75.0, 89.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_matches_analytic_fresnel(self, bare_stack, angle_deg, polarization):
        wavelength = 550.0
        result = solve_polarization(bare_stack, wavelength, angle_deg, polarization)
        r_ref, t_ref = fresnel_rt(1.0, 1.52, angle_deg, polarization)
        assert result.reflectance == pytest.approx(r_ref, abs=TIGHT)
        assert result.transmittance == pytest.approx(t_ref, abs=TIGHT)

    def test_wavelength_independent(self, bare_stack):
        # A bare lossless interface has no wavelength dependence; the matrix
        # solver must reproduce that (no phantom spectral structure).
        for wavelength in (400.0, 550.0, 900.0):
            result = solve_polarization(bare_stack, wavelength, 0.0, "s")
            r_ref, _ = fresnel_rt(1.0, 1.52, 0.0, "s")
            assert result.reflectance == pytest.approx(r_ref, abs=TIGHT)


class TestQuarterWaveAR:
    """The demo relation: quarter-wave AR layer beats the bare substrate."""

    def test_design_wavelength_reflectance_well_below_bare(
        self, quarter_wave_stack, bare_stack, demo_design_wavelength
    ):
        coated = solve_polarization(quarter_wave_stack, demo_design_wavelength, 0.0, "s")
        bare = solve_polarization(bare_stack, demo_design_wavelength, 0.0, "s")
        assert coated.reflectance < 0.5 * bare.reflectance
        assert coated.reflectance < 0.02  # ~1.26 % for the demo materials
        assert bare.reflectance == pytest.approx(0.0426, abs=1e-3)

    def test_matches_closed_form(self, quarter_wave_stack, demo_design_wavelength):
        # Single quarter-wave layer at normal incidence:
        # R = ((n0*ns - n1^2) / (n0*ns + n1^2))^2
        n0, n1, ns = 1.0, 1.38, 1.52
        expected = ((n0 * ns - n1 * n1) / (n0 * ns + n1 * n1)) ** 2
        result = solve_polarization(quarter_wave_stack, demo_design_wavelength, 0.0, "s")
        assert result.reflectance == pytest.approx(expected, abs=TIGHT)


class TestHalfWaveLayer:
    """Half-wave layer at the design wavelength: AR effect gone, R back to bare."""

    def test_half_wave_is_absentee_layer(
        self, half_wave_stack, bare_stack, demo_design_wavelength
    ):
        coated = solve_polarization(half_wave_stack, demo_design_wavelength, 0.0, "s")
        bare = solve_polarization(bare_stack, demo_design_wavelength, 0.0, "s")
        assert coated.reflectance == pytest.approx(bare.reflectance, abs=LOOSE)

    def test_half_wave_much_worse_than_quarter_wave(
        self, half_wave_stack, quarter_wave_stack, demo_design_wavelength
    ):
        half = solve_polarization(half_wave_stack, demo_design_wavelength, 0.0, "s")
        quarter = solve_polarization(quarter_wave_stack, demo_design_wavelength, 0.0, "s")
        assert half.reflectance > 3.0 * quarter.reflectance


class TestGrazingIncidence:
    """Reflectance must increase as the angle of incidence approaches grazing."""

    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_bare_interface_rises_towards_grazing(self, bare_stack, polarization):
        angles = [0.0, 30.0, 60.0, 80.0]
        values = [
            solve_polarization(bare_stack, 550.0, a, polarization).reflectance for a in angles
        ]
        # s rises monotonically from normal incidence; p does too once past
        # the Brewster angle (~56.7 deg for air/glass) — compare 0 vs 80 deg.
        assert values[-1] > values[0]
        assert values[-1] > values[1]

    def test_s_polarization_monotonic(self, bare_stack):
        angles = [0.0, 20.0, 40.0, 60.0, 80.0]
        values = [solve_polarization(bare_stack, 550.0, a, "s").reflectance for a in angles]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_coated_stack_rises_towards_grazing(self, quarter_wave_stack):
        normal = solve_polarization(quarter_wave_stack, 550.0, 0.0, "s").reflectance
        grazing = solve_polarization(quarter_wave_stack, 550.0, 85.0, "s").reflectance
        assert grazing > normal


class TestObliqueIncidenceDipShift:
    """End-to-end lock on the cos(theta) factor inside the phase thickness.

    A layer with quarter-wave optical thickness at lambda0 (normal incidence)
    keeps its reflectance minimum where the *normal* optical path is a
    quarter wave, i.e. at lambda0 * cos(theta_layer).  If the cos(theta)
    factor were missing from the phase thickness, the dip would stay glued to
    lambda0 and this test would fail.
    """

    def _dip_wavelength(self, stack, angle_deg):
        spectrum = scan_reflectance(stack, 400.0, 700.0, 601, angle_deg, "s")
        return min(spectrum, key=lambda p: p.result.reflectance).wavelength

    def test_dip_shifts_to_lambda0_cos_theta(self, quarter_wave_stack, demo_design_wavelength):
        angle_deg = 45.0
        sin_layer = math.sin(math.radians(angle_deg)) / 1.38
        cos_layer = math.sqrt(1.0 - sin_layer * sin_layer)
        expected = demo_design_wavelength * cos_layer

        dip_normal = self._dip_wavelength(quarter_wave_stack, 0.0)
        dip_oblique = self._dip_wavelength(quarter_wave_stack, angle_deg)

        assert dip_normal == pytest.approx(demo_design_wavelength, rel=0.01)
        assert dip_oblique == pytest.approx(expected, rel=0.01)
        # And it really moved: the shift is ~14 % of lambda0, far above the
        # 0.5 nm sampling step.
        assert dip_normal - dip_oblique > 0.10 * demo_design_wavelength


class TestSpectrumConsistency:
    """Spectrum points must be the very same solver truth, point by point."""

    def test_scan_matches_single_point_solver(self, quarter_wave_stack):
        angle_deg, polarization = 30.0, "p"
        spectrum = scan_reflectance(quarter_wave_stack, 480.0, 620.0, 8, angle_deg, polarization)
        for point in spectrum:
            direct = solve_polarization(
                quarter_wave_stack, point.wavelength, angle_deg, polarization
            )
            assert point.result.reflectance == pytest.approx(direct.reflectance, abs=TIGHT)
            assert point.result.transmittance == pytest.approx(direct.transmittance, abs=TIGHT)

    def test_scan_endpoints_included(self, quarter_wave_stack):
        spectrum = scan_reflectance(quarter_wave_stack, 400.0, 700.0, 301, 0.0, "s")
        assert len(spectrum) == 301
        assert spectrum[0].wavelength == pytest.approx(400.0)
        assert spectrum[-1].wavelength == pytest.approx(700.0)
