"""HTTP-layer tests: endpoints, validation errors, demo stack, concurrency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from thinopt import create_app

from conftest import LOOSE, fresnel_rt


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _demo_request(**overrides):
    payload = {
        "incident": 1.0,
        "substrate": 1.52,
        "layers": [{"index": 1.38, "thickness": 550.0 / (4.0 * 1.38)}],
        "wavelength": 550.0,
        "angle_deg": 0.0,
        "polarization": "s",
    }
    payload.update(overrides)
    return payload


class TestReflectanceEndpoint:
    def test_single_point_quarter_wave(self, client):
        response = client.post("/api/v1/reflectance", json=_demo_request())
        assert response.status_code == 200
        body = response.get_json()
        assert body["reflectance"] == pytest.approx(0.0126, abs=1e-3)
        assert body["reflectance"] + body["transmittance"] == pytest.approx(1.0, abs=LOOSE)
        assert body["absorptance"] == pytest.approx(0.0, abs=LOOSE)
        # Both polarisation components are always reported.
        assert set(body["components"]) == {"s", "p"}

    def test_bare_interface_matches_fresnel(self, client):
        request = _demo_request(layers=[], angle_deg=37.0, polarization="p")
        body = client.post("/api/v1/reflectance", json=request).get_json()
        r_ref, t_ref = fresnel_rt(1.0, 1.52, 37.0, "p")
        assert body["reflectance"] == pytest.approx(r_ref, abs=1e-12)
        assert body["transmittance"] == pytest.approx(t_ref, abs=1e-12)

    def test_complex_index_accepted(self, client):
        request = _demo_request(layers=[{"index": {"re": 1.7, "im": 0.4}, "thickness": 180.0}])
        response = client.post("/api/v1/reflectance", json=request)
        assert response.status_code == 200
        body = response.get_json()
        assert body["absorptance"] > 0.0
        total = body["reflectance"] + body["transmittance"] + body["absorptance"]
        assert total == pytest.approx(1.0, abs=LOOSE)

    def test_avg_polarization_is_mean_of_components(self, client):
        body = client.post(
            "/api/v1/reflectance", json=_demo_request(angle_deg=50.0, polarization="avg")
        ).get_json()
        mean_r = 0.5 * (
            body["components"]["s"]["reflectance"] + body["components"]["p"]["reflectance"]
        )
        assert body["reflectance"] == pytest.approx(mean_r, abs=1e-15)


class TestSpectrumEndpoint:
    def test_spectrum_shape_and_values(self, client):
        request = _demo_request(wavelength=None)
        request.pop("wavelength")
        request.update({"wavelength_start": 400.0, "wavelength_stop": 700.0, "points": 301})
        response = client.post("/api/v1/spectrum", json=request)
        assert response.status_code == 200
        body = response.get_json()
        points = body["points"]
        assert len(points) == 301
        assert points[0]["wavelength"] == pytest.approx(400.0)
        assert points[-1]["wavelength"] == pytest.approx(700.0)
        # The AR dip sits at the design wavelength: reflectance there is the
        # minimum of the scan and far below the bare-interface value.
        dip = min(points, key=lambda p: p["reflectance"])
        assert dip["wavelength"] == pytest.approx(550.0, abs=2.0)
        assert dip["reflectance"] < 0.02
        assert max(p["reflectance"] for p in points) > dip["reflectance"]

    def test_spectrum_points_are_solver_truth(self, client):
        """Spot-check scan points against the single-point endpoint."""
        scan_request = _demo_request(angle_deg=25.0, polarization="s")
        scan_request.pop("wavelength")
        scan_request.update({"wavelength_start": 500.0, "wavelength_stop": 600.0, "points": 11})
        points = client.post("/api/v1/spectrum", json=scan_request).get_json()["points"]
        for point in points:
            single = client.post(
                "/api/v1/reflectance",
                json=_demo_request(
                    wavelength=point["wavelength"], angle_deg=25.0, polarization="s"
                ),
            ).get_json()
            assert point["reflectance"] == pytest.approx(single["reflectance"], abs=1e-15)


class TestExampleEndpoint:
    def test_demo_relation_holds(self, client):
        body = client.get("/api/v1/example").get_json()
        coated = body["coated"]["reflectance"]
        bare = body["bare"]["reflectance"]
        assert coated < bare
        assert body["expected"]["holds"] is True
        # The advertised request payloads reproduce the advertised numbers.
        replay = client.post("/api/v1/reflectance", json=body["coated"]["request"]).get_json()
        assert replay["reflectance"] == pytest.approx(coated, abs=1e-15)


class TestValidation:
    """Illegal inputs must be rejected with structured 400 errors, pre-compute."""

    @pytest.mark.parametrize(
        "override, field",
        [
            ({"layers": [{"index": 1.38, "thickness": -10.0}]}, "layers[0].thickness"),
            ({"layers": [{"index": 0.0, "thickness": 10.0}]}, "layers[0].index"),
            ({"layers": [{"index": {"re": -1.2, "im": 0.1}, "thickness": 10.0}]}, "layers[0].index"),
            ({"substrate": {"re": 0.0, "im": 3.0}}, "substrate"),
            ({"wavelength": 0.0}, "wavelength"),
            ({"wavelength": -550.0}, "wavelength"),
            ({"angle_deg": 90.0}, "angle_deg"),
            ({"angle_deg": 120.0}, "angle_deg"),
            ({"angle_deg": -5.0}, "angle_deg"),
            ({"polarization": "circular"}, "polarization"),
        ],
    )
    def test_invalid_inputs_rejected(self, client, override, field):
        response = client.post("/api/v1/reflectance", json=_demo_request(**override))
        assert response.status_code == 400
        error = response.get_json()["error"]
        assert error["type"] == "validation_error"
        assert any(d["field"] == field for d in error["details"])

    def test_multiple_errors_reported_together(self, client):
        response = client.post(
            "/api/v1/reflectance",
            json=_demo_request(wavelength=-1.0, angle_deg=95.0),
        )
        fields = {d["field"] for d in response.get_json()["error"]["details"]}
        assert {"wavelength", "angle_deg"} <= fields

    def test_non_object_body_rejected(self, client):
        response = client.post("/api/v1/reflectance", json=[1, 2, 3])
        assert response.status_code == 400
        assert response.get_json()["error"]["type"] == "validation_error"

    def test_malformed_json_rejected(self, client):
        response = client.post(
            "/api/v1/reflectance", data="{not json", content_type="application/json"
        )
        assert response.status_code == 400

    def test_spectrum_range_validated(self, client):
        request = _demo_request()
        request.pop("wavelength")
        request.update({"wavelength_start": 700.0, "wavelength_stop": 400.0})
        response = client.post("/api/v1/spectrum", json=request)
        assert response.status_code == 400

    def test_error_body_is_json(self, client):
        response = client.post("/api/v1/reflectance", json=_demo_request(wavelength=-1.0))
        assert response.content_type.startswith("application/json")


class TestConcurrency:
    """Concurrent requests with different stacks must not interfere."""

    def test_parallel_requests_isolated(self, client):
        app = create_app()

        def work(tag: int):
            # Each thread gets its own client, its own stack parameters and
            # its own expected reflectance; nothing is shared but the app.
            thread_client = app.test_client()
            thickness = 80.0 + tag
            reflectances = set()
            for _ in range(5):
                body = thread_client.post(
                    "/api/v1/reflectance",
                    json=_demo_request(
                        layers=[{"index": 1.38, "thickness": thickness}],
                        wavelength=500.0 + tag,
                        angle_deg=float(tag),
                    ),
                ).get_json()
                assert body["wavelength"] == pytest.approx(500.0 + tag)
                assert 0.0 <= body["reflectance"] <= 1.0
                reflectances.add(body["reflectance"])
            # Same input -> identical output on every retry (no cross-talk).
            assert len(reflectances) == 1
            return reflectances.pop()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, range(16)))
        # Distinct stacks genuinely produced distinct reflectances.
        assert len(set(results)) > 1
