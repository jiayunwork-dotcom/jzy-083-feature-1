"""Unit tests for the optics primitives: admittances and phase thickness."""

from __future__ import annotations

import cmath
import math

import pytest

from thinopt.optics import (
    cos_theta_in_layer,
    effective_admittance,
    phase_thickness,
)


class TestEffectiveAdmittance:
    """s and p admittances are independent expressions and must not be mixed."""

    def test_normal_incidence_both_reduce_to_index(self):
        n = 1.5
        cos_t = 1.0
        assert effective_admittance(n, cos_t, "s") == pytest.approx(n)
        assert effective_admittance(n, cos_t, "p") == pytest.approx(n)

    def test_oblique_s_and_p_differ(self):
        n, cos_t = 1.5, 0.8
        eta_s = effective_admittance(n, cos_t, "s")
        eta_p = effective_admittance(n, cos_t, "p")
        assert eta_s == pytest.approx(n * cos_t)
        assert eta_p == pytest.approx(n / cos_t)
        assert eta_s != pytest.approx(eta_p)

    def test_complex_index_supported(self):
        n = complex(0.18, 2.4)  # metal-like
        cos_t = 0.9 + 0.05j
        assert effective_admittance(n, cos_t, "s") == pytest.approx(n * cos_t)
        assert effective_admittance(n, cos_t, "p") == pytest.approx(n / cos_t)

    def test_unknown_polarization_rejected(self):
        with pytest.raises(ValueError):
            effective_admittance(1.5, 1.0, "x")


class TestCosThetaInLayer:
    def test_normal_incidence_is_one(self):
        assert cos_theta_in_layer(1.5, 0.0, 1.0) == pytest.approx(1.0)

    def test_matches_snells_law(self):
        n0, n1, theta0 = 1.0, 1.5, math.radians(40.0)
        cos_t = cos_theta_in_layer(n1, math.sin(theta0), n0)
        sin_t = n0 * math.sin(theta0) / n1
        assert cos_t == pytest.approx(math.sqrt(1.0 - sin_t**2))

    def test_absorbing_layer_branch_decays(self):
        # Passive medium: the field must decay towards +z, i.e. Im(N cos) >= 0.
        n = complex(1.5, 0.5)
        cos_t = cos_theta_in_layer(n, math.sin(math.radians(60.0)), 1.0)
        assert (n * cos_t).imag >= 0.0
        assert cos_t.real >= 0.0


class TestPhaseThickness:
    def test_quarter_wave_gives_pi_over_2(self):
        n, d, wavelength = 1.38, 550.0 / (4.0 * 1.38), 550.0
        delta = phase_thickness(n, d, 1.0, wavelength)
        assert delta == pytest.approx(math.pi / 2.0)

    def test_cos_theta_factor_present(self):
        """The oblique-incidence cos(theta) factor must multiply the phase.

        This is the classic bug: dropping cos(theta) leaves the normal-
        incidence phase in place and shifts every spectral feature at
        oblique incidence.  Guard the factor directly.
        """
        n, d, wavelength = 1.5, 100.0, 550.0
        cos_t = 0.6
        delta = phase_thickness(n, d, cos_t, wavelength)
        expected = 2.0 * math.pi * n * d * cos_t / wavelength
        without_cos = 2.0 * math.pi * n * d / wavelength
        assert delta == pytest.approx(expected)
        assert abs(delta - without_cos) > 0.1 * abs(without_cos)

    def test_complex_index_gives_complex_phase(self):
        delta = phase_thickness(complex(1.5, 0.3), 100.0, 0.9, 550.0)
        assert delta.imag != 0.0
