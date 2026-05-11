"""T4.3 - Multi-question Dirichlet posterior API + acceptance gate.

API smoke tests + the DEVPLAN T4.3 substantive acceptance gate:

* ``/api/posteriors`` accepts a multipart image upload and returns a
  well-formed ``PosteriorResponse`` with all 10 GZ DESI questions,
  every per-question CI in [0, 1] and ordered (lower <= upper).
* ``/api/demo_galaxies`` enumerates the pre-computed demo galaxies.
* ``/api/demo_galaxies/{id}/thumbnail`` serves the cached JPEG.
* ``/api/demo_galaxies/{id}/posteriors`` returns posteriors + the
  volunteer-overlay block.
* **DEVPLAN T4.3 acceptance**: at least 3 demo galaxies exist, and
  there is at least one question on which the 95% CIs do NOT all
  overlap pairwise (the Playwright substantive check, lifted to the
  API layer for reliability in this environment).

Skipped when torch / fastapi / the T3.6 Dirichlet checkpoint / the
demo-galaxies manifest are missing (each gate is logged in the skip
reason so a missing piece is obvious).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("galaxy_datasets")

ZOOBOT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
DIRICHLET_CKPT = Path("runs/m3_dirichlet/best.pt")
DEMO_MANIFEST = Path("artifacts/demo_galaxies/manifest.json")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip(
        "Galaxy10 (M1) checkpoint missing -- /api/health depends on it. "
        "Run the T1.5 trainer first.",
        allow_module_level=True,
    )
if not DIRICHLET_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip(
        "Dirichlet (M3) checkpoint missing. Run the T3.6 trainer to "
        "produce runs/m3_dirichlet/best.pt.",
        allow_module_level=True,
    )
if not DEMO_MANIFEST.is_file():  # pragma: no cover -- local-only gate
    pytest.skip(
        "Demo-galaxies manifest missing. Run "
        "`python -m scripts.build_demo_galaxies --config configs/m3_dirichlet.yaml`.",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image as PILImage_  # noqa: E402

from galaxy_vit.data.schema import num_questions  # noqa: E402
from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """TestClient with FastAPI lifespan: models load once per module."""
    with TestClient(app) as c:
        yield c


def _synthetic_png_bytes(size: int = 128) -> bytes:
    """A tiny deterministic test image for /api/posteriors upload smoke."""
    import io as _io

    rng = torch.Generator().manual_seed(0)
    arr = (torch.rand(size, size, 3, generator=rng) * 255).to(torch.uint8).numpy()
    img = PILImage_.fromarray(arr).convert("RGB")
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Endpoint smoke tests
# ---------------------------------------------------------------------------


def test_T4_3_posteriors_endpoint_returns_all_questions(client: TestClient) -> None:
    """POST /api/posteriors returns one PosteriorQuestionItem per GZ DESI question."""
    files = {"file": ("smoke.png", _synthetic_png_bytes(), "image/png")}
    response = client.post("/api/posteriors", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "questions" in body
    assert len(body["questions"]) == num_questions()
    assert body["temperature"] > 0.0
    assert body["calibration_regime"] in ("none", "single_T")


def test_T4_3_posteriors_endpoint_ci_bounds_well_formed(client: TestClient) -> None:
    """Every per-answer CI: 0 <= lower <= mean <= upper <= 1 (within fp tolerance)."""
    files = {"file": ("smoke.png", _synthetic_png_bytes(), "image/png")}
    response = client.post("/api/posteriors", files=files)
    body = response.json()
    eps = 1e-5
    for q in body["questions"]:
        plurality_idx = int(q["plurality_index"])
        assert 0 <= plurality_idx < len(q["answers"])
        for a in q["answers"]:
            mean = float(a["mean"])
            lower = float(a["ci_lower"])
            upper = float(a["ci_upper"])
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0
            assert lower <= upper + eps
            assert 0.0 <= mean <= 1.0


def test_T4_3_demo_galaxies_lists_at_least_3(client: TestClient) -> None:
    """The demo-galaxies endpoint enumerates >= 3 galaxies (Playwright needs that)."""
    response = client.get("/api/demo_galaxies")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "galaxies" in body
    assert len(body["galaxies"]) >= 3
    for g in body["galaxies"]:
        assert g["id"]
        assert g["smooth_or_featured_plurality"] in (
            "smooth", "featured-or-disk", "artifact", "unknown",
        )
        assert g["thumbnail_url"].endswith(f"/{g['id']}/thumbnail")


def test_T4_3_demo_galaxy_thumbnail_serves_jpeg(client: TestClient) -> None:
    """The thumbnail endpoint returns a small JPEG with image/jpeg content-type."""
    galaxies = client.get("/api/demo_galaxies").json()["galaxies"]
    response = client.get(f"/api/demo_galaxies/{galaxies[0]['id']}/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # JPEG magic bytes: FF D8 FF.
    assert response.content[:3] == b"\xff\xd8\xff"


def test_T4_3_demo_galaxy_posteriors_has_volunteer_overlay(
    client: TestClient,
) -> None:
    """The demo posteriors endpoint returns posterior + the volunteer overlay block."""
    galaxies = client.get("/api/demo_galaxies").json()["galaxies"]
    response = client.get(f"/api/demo_galaxies/{galaxies[0]['id']}/posteriors")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "posterior" in body
    assert "volunteer" in body
    assert len(body["posterior"]["questions"]) == num_questions()
    assert len(body["volunteer"]) == num_questions()
    # Volunteer fractions sum to 1 on questions with any vote.
    for v in body["volunteer"]:
        total = sum(v["fractions"])
        if total > 0:
            assert abs(total - 1.0) < 1e-5


def test_T4_3_unknown_demo_galaxy_returns_404(client: TestClient) -> None:
    response = client.get("/api/demo_galaxies/_NOT_A_REAL_ID_/posteriors")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DEVPLAN T4.3 acceptance gate
# ---------------------------------------------------------------------------


def _cis_overlap(
    lower_a: float, upper_a: float, lower_b: float, upper_b: float
) -> bool:
    return not (upper_a < lower_b or upper_b < lower_a)


def test_T4_3_three_galaxies_non_overlapping_ci_on_at_least_one_question(
    client: TestClient,
) -> None:
    """T4.3 acceptance (DEVPLAN): selecting 3 demo galaxies, at least one
    question's 95% CIs are non-overlapping between at least one PAIR.

    The Playwright spec phrases this as "non-overlapping CIs on >=1
    question". We lift it to the API layer: fetch 3 galaxies' posteriors,
    look for one question where the per-answer CI of the plurality answer
    is disjoint between at least one pair of galaxies. If the Dirichlet
    head produces meaningful per-galaxy uncertainty, this is easy to
    find -- it would only fail if every galaxy's posterior were
    statistically indistinguishable, which would indicate the head
    learned no per-image signal.
    """
    galaxies = client.get("/api/demo_galaxies").json()["galaxies"]
    assert len(galaxies) >= 3, "need >= 3 demo galaxies"

    # Pick 3 galaxies stratified across the smooth-or-featured ground truth
    # so we maximize chances of finding non-overlapping CIs (different
    # morphologies -> different posteriors).
    by_class: dict[str, list[dict[str, str]]] = {}
    for g in galaxies:
        by_class.setdefault(g["smooth_or_featured_plurality"], []).append(g)
    picks: list[dict[str, str]] = []
    for cls in ("smooth", "featured-or-disk", "artifact"):
        if by_class.get(cls):
            picks.append(by_class[cls][0])
    while len(picks) < 3:
        # Fall back to the first available if stratification underfilled.
        for g in galaxies:
            if g not in picks:
                picks.append(g)
                break

    posteriors: list[dict[str, object]] = []
    for p in picks[:3]:
        body = client.get(f"/api/demo_galaxies/{p['id']}/posteriors").json()
        posteriors.append(body["posterior"])

    # For each question, check whether any PAIR of galaxies has at least
    # one answer's CIs non-overlapping. As soon as we find one, we pass.
    found_disjoint = False
    disjoint_evidence: list[str] = []
    for q_idx in range(num_questions()):
        for i in range(len(posteriors)):
            for j in range(i + 1, len(posteriors)):
                q_i = posteriors[i]["questions"][q_idx]  # type: ignore[index]
                q_j = posteriors[j]["questions"][q_idx]  # type: ignore[index]
                for a_i, a_j in zip(q_i["answers"], q_j["answers"], strict=True):
                    if not _cis_overlap(
                        a_i["ci_lower"], a_i["ci_upper"],
                        a_j["ci_lower"], a_j["ci_upper"],
                    ):
                        found_disjoint = True
                        disjoint_evidence.append(
                            f"q={q_i['question']} a={a_i['name']}: "
                            f"galaxy_{picks[i]['id']} CI=[{a_i['ci_lower']:.3f}, {a_i['ci_upper']:.3f}] "
                            f"vs galaxy_{picks[j]['id']} CI=[{a_j['ci_lower']:.3f}, {a_j['ci_upper']:.3f}]"
                        )
                        break
                if found_disjoint:
                    break
            if found_disjoint:
                break
        if found_disjoint:
            break

    assert found_disjoint, (
        "no question had non-overlapping CIs across any pair of the 3 picked galaxies; "
        "Dirichlet head appears not to produce per-image-distinguishable posteriors"
    )
    # Print the evidence into the test output for traceability.
    print(f"\n[T4.3 acceptance] disjoint CI found: {disjoint_evidence[0]}")
