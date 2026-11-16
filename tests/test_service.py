import pandas as pd
import pytest

from jobmatch import service
from jobmatch.neural import ROOT

needs_model = pytest.mark.skipif(not (ROOT / ".hf-cache").exists(), reason="model not downloaded")


def test_plain_resume_sections_reordered():
    text = "Staff Accountant\nSummary\nDetail oriented.\nExperience\nClosed the books monthly.\nSkills\nExcel, GL"
    assert service.plain_resume_embed_text(text) == "Staff Accountant . Excel, GL . Detail oriented. . Closed the books monthly."


def test_plain_resume_without_headings_falls_back_to_text():
    assert service.plain_resume_embed_text("line cook   with grill experience") == "line cook with grill experience"


@needs_model
def test_rank_endpoint_end_to_end(tmp_path, monkeypatch):
    import faiss
    from fastapi.testclient import TestClient
    from jobmatch.lexical import job_keywords
    from jobmatch.neural import BiEncoder, build_index

    jobs = pd.DataFrame({
        "job_id": ["a", "b", "c"],
        "title": ["Staff Accountant", "Line Cook", "Registered Nurse"],
        "embed_text": ["Staff Accountant . general ledger, reconciliations, Excel",
                       "Line Cook . grill, prep, food safety", "Registered Nurse . patient care, ICU"],
    })
    jobs["keywords"] = [job_keywords(t, "", e) for t, e in zip(jobs.title, jobs.embed_text)]
    jobs.to_parquet(tmp_path / "jobs.parquet")
    faiss.write_index(build_index(BiEncoder(device="cpu").encode(jobs.embed_text.tolist())), str(tmp_path / "jobs.faiss"))
    monkeypatch.setattr(service, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(service, "RERANKER", str(tmp_path / "no-reranker"))

    with TestClient(service.app) as client:
        r = client.post("/rank", json={"resume_text": "Accountant\nSkills\nreconciliations, Excel, ledger", "k": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["results"][0]["job_id"] == "a"
        assert "reconciliations" in body["results"][0]["matched_skills"]
        assert len(body["results"]) == 2 and body["timings_ms"]["total"] > 0
        assert client.post("/rank", json={"resume_text": "x", "rerank": True}).status_code == 503
        assert client.post("/rank", json={"resume_text": ""}).status_code == 422
