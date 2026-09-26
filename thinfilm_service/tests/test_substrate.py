"""Physics benchmarks for the finite-substrate (incoherent back-face) channel.

Each test pins one of the correctness criteria the lab agreed on:

* degenerate self-consistency: a lossless substrate whose back medium has
  the substrate's own index (the back interface vanishes, i.e. the
  semi-infinite-substrate limit) must reproduce the front-only coherent
  result exactly — the two code paths are the same curve in that limit;
* energy conservation: after the incoherent summation the whole sample
  still satisfies R + T + A = 1 with A >= 0;
* the back face genuinely contributes: with a good AR coating the total
  reflectance clearly exceeds the front-only value, by an amount set by
  the bare back-interface reflectance;
* series convergence: the closed-form geometric sum matches an explicit
  bounce-by-bounce accumulation, and the bounce contributions decay
  monotonically to zero as absorption/thickness grow.
"""

from __future__ import annotations

import math

import pytest

from thinopt.matrices import Layer, Stack
from thinopt.solver import solve_polarization, solve_polarization_sine
from thinopt.substrate import (
    FiniteSubstrate,
    incoherent_cascade,
    single_pass_transmittance,
    solve_slab,
    solve_slab_polarization,
)

from conftest import LOOSE, TIGHT, fresnel_rt

#: A standard 0.5 mm glass slide, in nm (the wavelength unit of the suite).
SLIDE_THICKNESS = 5.0e5


def _slab(back_index=1.0, thickness=SLIDE_THICKNESS) -> FiniteSubstrate:
    return FiniteSubstrate(thickness=thickness, back_index=back_index)


class TestDegenerateLimit:
    """Lossless slab + back index == substrate index <=> semi-infinite substrate."""

    @pytest.mark.parametrize("wavelength", [400.0, 550.0, 700.0])
    @pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_vanishing_back_interface_recovers_front_only(
        self, quarter_wave_stack, wavelength, angle_deg, polarization
    ):
        substrate = _slab(back_index=quarter_wave_stack.substrate_index)
        whole = solve_slab_polarization(
            quarter_wave_stack, substrate, wavelength, angle_deg, polarization
        )
        front = solve_polarization(quarter_wave_stack, wavelength, angle_deg, polarization)
        # The back interface has vanished: no attenuation, no back reflection.
        assert whole.single_pass_transmittance == pytest.approx(1.0)
        assert whole.back.reflectance == pytest.approx(0.0, abs=TIGHT)
        # And the totals fall exactly back onto the front-only coherent curve.
        assert whole.reflectance == pytest.approx(front.reflectance, abs=TIGHT)
        assert whole.transmittance == pytest.approx(front.transmittance, abs=TIGHT)
        assert whole.absorptance == pytest.approx(front.absorptance, abs=TIGHT)

    def test_degenerate_with_absorbing_layers(self):
        """Absorbing *layers* must not disturb the limit (the slab stays lossless)."""
        stack = Stack(
            incident_index=1.0,
            substrate_index=1.52,
            layers=(
                Layer(index=complex(1.7, 0.4), thickness=180.0),
                Layer(index=1.38, thickness=95.0),
            ),
        )
        substrate = _slab(back_index=1.52)
        whole = solve_slab_polarization(stack, substrate, 550.0, 40.0, "p")
        front = solve_polarization(stack, 550.0, 40.0, "p")
        assert whole.reflectance == pytest.approx(front.reflectance, abs=TIGHT)
        assert whole.transmittance == pytest.approx(front.transmittance, abs=TIGHT)
        assert whole.absorptance == pytest.approx(front.absorptance, abs=TIGHT)

    def test_sine_entry_point_is_the_same_solver(self, quarter_wave_stack):
        """The sine-parameterised entry point is not a second algorithm."""
        angle_deg = 33.0
        by_angle = solve_polarization(quarter_wave_stack, 550.0, angle_deg, "s")
        by_sine = solve_polarization_sine(
            quarter_wave_stack, 550.0, math.sin(math.radians(angle_deg)), "s"
        )
        assert by_angle.reflectance == by_sine.reflectance
        assert by_angle.transmittance == by_sine.transmittance
        assert by_angle.r == by_sine.r


class TestEnergyConservation:
    """Whole sample: R + T + A == 1, A >= 0, with and without absorption."""

    @pytest.mark.parametrize("substrate_k", [0.0, 1e-4, 1e-2])
    @pytest.mark.parametrize("angle_deg", [0.0, 45.0, 70.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_energy_balance_with_absorbing_substrate(
        self, substrate_k, angle_deg, polarization
    ):
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, substrate_k),
            layers=(
                Layer(index=complex(1.7, 0.4), thickness=180.0),
                Layer(index=1.38, thickness=95.0),
            ),
        )
        whole = solve_slab_polarization(
            stack, _slab(thickness=2.0e5), 550.0, angle_deg, polarization
        )
        total = whole.reflectance + whole.transmittance + whole.absorptance
        assert total == pytest.approx(1.0, abs=LOOSE)
        assert whole.absorptance >= 0.0
        assert 0.0 <= whole.reflectance <= 1.0
        assert 0.0 <= whole.transmittance <= 1.0

    def test_unpolarised_energy_balance(self, quarter_wave_stack):
        point = solve_slab(quarter_wave_stack, _slab(), 550.0, 50.0)
        avg = point.averaged()
        assert avg.reflectance + avg.transmittance + avg.absorptance == pytest.approx(
            1.0, abs=LOOSE
        )

    def test_avg_is_mean_of_components(self, quarter_wave_stack):
        point = solve_slab(quarter_wave_stack, _slab(), 550.0, 50.0)
        avg = point.averaged()
        assert avg.reflectance == pytest.approx(
            0.5 * (point.s.reflectance + point.p.reflectance), abs=1e-15
        )
        assert avg.front.reflectance == pytest.approx(
            0.5 * (point.s.front.reflectance + point.p.front.reflectance), abs=1e-15
        )


class TestBackFaceContribution:
    """With a good AR coating the bare back face dominates the measurement."""

    def test_ar_coated_slide_reflectance_is_back_dominated(self, quarter_wave_stack):
        whole = solve_slab_polarization(quarter_wave_stack, _slab(), 550.0, 0.0, "s")
        front_r = whole.front.reflectance
        back_r = whole.back.reflectance  # bare glass/air interface
        assert front_r < 0.02  # the AR coating works
        assert back_r == pytest.approx(0.0426, abs=1e-3)
        # The whole-sample value clearly exceeds the front-only one...
        assert whole.reflectance > 3.0 * front_r
        # ...and the excess is the bare back-interface reflectance, scaled by
        # the front stack's two-way transmission (alpha = 1, q ~ 5e-4).
        excess = whole.reflectance - front_r
        assert excess == pytest.approx(
            back_r * whole.front.transmittance * whole.reverse.transmittance,
            rel=2e-3,
        )
        assert excess == pytest.approx(back_r, rel=0.1)  # right order of magnitude

    def test_bare_slide_matches_textbook_incoherent_formula(self, bare_stack):
        """Uncoated lossless slide in air: R = 2r/(1+r), T = (1-r)/(1+r)."""
        whole = solve_slab_polarization(bare_stack, _slab(), 550.0, 0.0, "s")
        r, _ = fresnel_rt(1.0, 1.52, 0.0, "s")
        assert whole.reflectance == pytest.approx(2.0 * r / (1.0 + r), abs=TIGHT)
        assert whole.transmittance == pytest.approx((1.0 - r) / (1.0 + r), abs=TIGHT)

    @pytest.mark.parametrize("angle_deg", [20.0, 40.0, 55.0])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    def test_bare_slide_at_oblique_incidence(self, bare_stack, angle_deg, polarization):
        """The same closed form holds at angle: every interface sees the same r."""
        whole = solve_slab_polarization(bare_stack, _slab(), 550.0, angle_deg, polarization)
        r, _ = fresnel_rt(1.0, 1.52, angle_deg, polarization)
        assert whole.reflectance == pytest.approx(2.0 * r / (1.0 + r), abs=TIGHT)
        assert whole.transmittance == pytest.approx((1.0 - r) / (1.0 + r), abs=TIGHT)

    def test_reverse_transmittance_matches_forward_for_lossless_stack(
        self, quarter_wave_stack
    ):
        """Reciprocity: a lossless stack transmits identically from both sides.

        An independent lock on the reverse (substrate-side) solve — if the
        reversed layer order or the complex propagation sine were wrong,
        T'_f would drift away from T_f.
        """
        whole = solve_slab_polarization(quarter_wave_stack, _slab(), 550.0, 35.0, "s")
        assert whole.reverse.transmittance == pytest.approx(
            whole.front.transmittance, abs=LOOSE
        )


class TestSeriesConvergence:
    """The geometric bounce series: closed form vs explicit accumulation."""

    @staticmethod
    def _explicit_bounce_sum(whole, num_bounces):
        """Accumulate the substrate round trips one by one (the slow way)."""
        front, reverse, back = whole.front, whole.reverse, whole.back
        alpha = whole.single_pass_transmittance
        q = whole.round_trip_factor
        term_r = front.transmittance * reverse.transmittance * back.reflectance * alpha**2
        term_t = front.transmittance * back.transmittance * alpha
        r_extra, t_tail, terms = 0.0, 0.0, []
        for _ in range(num_bounces):
            if term_r == 0.0:  # geometric tail underflowed: nothing left to add
                break
            terms.append(term_r)
            r_extra += term_r
            t_tail += term_t
            term_r *= q
            term_t *= q
        return front.reflectance + r_extra, t_tail, terms

    def test_closed_form_matches_explicit_sum(self):
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 2e-5),
            layers=(Layer(index=1.38, thickness=99.6),),
        )
        whole = solve_slab_polarization(stack, _slab(thickness=3.0e5), 550.0, 25.0, "s")
        assert 0.0 < whole.round_trip_factor < 1.0
        r_sum, t_sum, terms = self._explicit_bounce_sum(whole, num_bounces=200)
        assert whole.reflectance == pytest.approx(r_sum, abs=TIGHT)
        assert whole.transmittance == pytest.approx(t_sum, abs=TIGHT)
        # Successive bounce contributions shrink geometrically towards zero.
        assert len(terms) > 5  # the series genuinely has multiple bounces
        assert all(b < a for a, b in zip(terms, terms[1:]))
        assert terms[-1] < 1e-12 * terms[0]

    def test_back_contribution_decays_monotonically_with_absorption(
        self, quarter_wave_stack
    ):
        """More absorption (or more thickness) => less light survives the
        round trips, monotonically, down to zero."""
        contributions = []
        for substrate_k in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
            stack = Stack(
                incident_index=1.0,
                substrate_index=complex(1.52, substrate_k),
                layers=quarter_wave_stack.layers,
            )
            whole = solve_slab_polarization(stack, _slab(), 550.0, 0.0, "s")
            contributions.append(whole.reflectance - whole.front.reflectance)
        assert all(b < a for a, b in zip(contributions, contributions[1:]))
        assert contributions[-1] == pytest.approx(0.0, abs=1e-12)

        # Same monotonic decay when the thickness grows at fixed absorption.
        contributions = []
        for thickness in (1.0e4, 1.0e5, 1.0e6, 1.0e7):
            stack = Stack(
                incident_index=1.0,
                substrate_index=complex(1.52, 1e-4),
                layers=quarter_wave_stack.layers,
            )
            whole = solve_slab_polarization(
                stack, _slab(thickness=thickness), 550.0, 0.0, "s"
            )
            contributions.append(whole.reflectance - whole.front.reflectance)
        assert all(b < a for a, b in zip(contributions, contributions[1:]))
        assert contributions[-1] == pytest.approx(0.0, abs=1e-12)

    def test_thick_absorbing_substrate_kills_the_back_channel(self, quarter_wave_stack):
        stack = Stack(
            incident_index=1.0,
            substrate_index=complex(1.52, 0.05),
            layers=quarter_wave_stack.layers,
        )
        whole = solve_slab_polarization(stack, _slab(thickness=1.0e6), 550.0, 0.0, "s")
        # Nothing crosses a millimetre of k = 0.05 material and comes back.
        assert whole.single_pass_transmittance == 0.0
        assert whole.round_trip_factor == 0.0
        assert whole.reflectance == pytest.approx(whole.front.reflectance, abs=TIGHT)
        assert whole.transmittance == 0.0
        assert whole.absorptance == pytest.approx(1.0 - whole.front.reflectance, abs=TIGHT)

    def test_single_pass_transmittance_follows_beer_lambert(self):
        # alpha = exp(-4 pi k d cos(theta) / lambda) at normal incidence.
        alpha = single_pass_transmittance(complex(1.5, 0.01), 1000.0, 1.0, 550.0)
        assert alpha == pytest.approx(math.exp(-4.0 * math.pi * 0.01 * 1000.0 / 550.0))
        assert single_pass_transmittance(complex(1.5, 0.0), 1.0e9, 1.0, 550.0) == 1.0

    def test_incoherent_cascade_energy_components(self):
        """Direct ingredient-level check of the closed-form cascade."""
        from thinopt.solver import PolarizationResult

        front = PolarizationResult(0.01, 0.97, 0.02, 0.1j, 0.9)
        reverse = PolarizationResult(0.03, 0.95, 0.02, 0.17j, 0.9)
        back = PolarizationResult(0.04, 0.96, 0.0, -0.2, 0.98)
        alpha = 0.9
        r, t, a = incoherent_cascade(front, reverse, back, alpha)
        q = 0.03 * 0.04 * 0.9**2
        assert r == pytest.approx(0.01 + 0.97 * 0.95 * 0.04 * 0.9**2 / (1.0 - q))
        assert t == pytest.approx(0.97 * 0.96 * 0.9 / (1.0 - q))
        assert r + t + a == pytest.approx(1.0, abs=LOOSE)

    def test_incoherent_cascade_trapped_light_limit(self):
        """q = 1 (lossless slab, no way out): the sample is just the front."""
        from thinopt.solver import PolarizationResult

        front = PolarizationResult(0.9, 0.0, 0.1, 0.9, 0.0)
        reverse = PolarizationResult(1.0, 0.0, 0.0, 1.0, 0.0)
        back = PolarizationResult(1.0, 0.0, 0.0, 1.0, 0.0)
        r, t, a = incoherent_cascade(front, reverse, back, 1.0)
        assert (r, t, a) == (0.9, 0.0, 0.1)

    def test_total_internal_reflection_at_back_face(self):
        """Back medium denser-side TIR: everything returns, T_total = 0."""
        # Incident medium 1.6 at 60 deg: the tangential wavevector exceeds
        # the back medium's index (1.0), so the back face totally reflects.
        stack = Stack(incident_index=1.6, substrate_index=1.5, layers=())
        whole = solve_slab_polarization(stack, _slab(), 550.0, 60.0, "s")
        assert whole.back.reflectance == pytest.approx(1.0, abs=TIGHT)
        assert whole.back.transmittance == pytest.approx(0.0, abs=TIGHT)
        # Lossless slab with a perfectly reflecting back: R_total = 1, T_total = 0.
        assert whole.reflectance == pytest.approx(1.0, abs=TIGHT)
        assert whole.transmittance == pytest.approx(0.0, abs=TIGHT)
        assert whole.absorptance == pytest.approx(0.0, abs=TIGHT)
