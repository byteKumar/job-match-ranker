"""Section-aware text cleaning for resumes and job postings.

Why sections: the embedding model (all-MiniLM-L6-v2) reads at most 256 word pieces, and a
median resume here is about 5,900 characters. If we feed raw text, the model sees only the
first few hundred words, often an objective statement. So we pull out the parts that say
what the person does (title, skills, recent experience) and put those first.
"""

from __future__ import annotations

import html as _html
import re

# Resume HTML (LiveCareer export) marks sections as <div class="section" id="SECTION_XXXX...">.
_SECTION_SPLIT = re.compile(r'<div class="section[^"]*" id="SECTION_([A-Z]{4})')
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# Anonymization placeholders left in the dataset; they carry no signal.
_PLACEHOLDERS = re.compile(r"\b(Company Name|City\s*,\s*State|Company Name\s*－)\b|－")

SKILL_SECTIONS = {"SKLL", "HILT", "TSKL", "CERT"}  # skills, highlights, technical skills, certifications
SUMMARY_SECTIONS = {"SUMM", "PRFL", "PRIN"}       # summary, profile, professional summary
EXPERIENCE_SECTIONS = {"EXPR", "WRKH"}             # experience, work history
EDUCATION_SECTIONS = {"EDUC"}


def normalize(text: str) -> str:
    text = _html.unescape(text or "")
    text = _PLACEHOLDERS.sub(" ", text)
    return _WS.sub(" ", text).strip()


def strip_tags(fragment: str) -> str:
    # Turn block boundaries into separators before dropping tags, so list items don't merge.
    fragment = re.sub(r"</(p|li|div|td|tr|br)\s*>|<br\s*/?>", " ; ", fragment)
    return normalize(_TAG.sub(" ", fragment)).replace(" ; ;", " ;").strip(" ;")


def resume_sections(resume_html: str) -> dict[str, str]:
    """Split resume HTML into {section_code: plain text}. Repeated codes are concatenated."""
    parts = _SECTION_SPLIT.split(resume_html or "")
    out: dict[str, str] = {}
    # parts = [preamble, code1, body1, code2, body2, ...]
    for code, body in zip(parts[1::2], parts[2::2]):
        body = body.split(">", 1)[-1]  # drop the rest of the opening <div ...> tag
        body = re.sub(r'<div class="sectiontitle"[^>]*>.*?</div>', " ", body, flags=re.S)
        txt = strip_tags(body)
        if txt:
            out[code] = (out.get(code, "") + " ; " + txt).strip(" ;")
    return out


def structure_resume(resume_html: str, resume_str: str) -> dict[str, str]:
    """Return title / summary / skills / experience / education / full fields for a resume."""
    secs = resume_sections(resume_html)
    pick = lambda codes: " ; ".join(secs[c] for c in secs if c in codes)
    title = secs.get("NAME", "") or normalize(resume_str)[:60]
    return {
        "title": title.strip(" ;"),
        "summary": pick(SUMMARY_SECTIONS),
        "skills": pick(SKILL_SECTIONS),
        "experience": pick(EXPERIENCE_SECTIONS),
        "education": pick(EDUCATION_SECTIONS),
        "full": normalize(resume_str),
    }


# Job descriptions are free text; requirement-style headings are common.
_REQ_HEADING = re.compile(
    r"(requirements|qualifications|what you(?:'|’)ll need|what you bring|skills|who you are|"
    r"minimum qualifications|preferred qualifications|experience required)\s*:?",
    re.I,
)


def structure_job(title: str, description: str, skills_desc: str = "") -> dict[str, str]:
    """Return title / requirements / description / full fields for a job posting."""
    desc = normalize(description)
    m = _REQ_HEADING.search(desc)
    requirements = desc[m.start(): m.start() + 1500] if m else ""
    skills = normalize(skills_desc)
    return {
        "title": normalize(title),
        "requirements": (skills + " ; " + requirements).strip(" ;") if skills else requirements,
        "description": desc,
        "full": normalize(title) + " . " + desc,
    }


def resume_embed_text(r: dict) -> str:
    """Section-ordered text for embedding: title, skills, summary, then experience."""
    return " . ".join(x for x in (r["title"], r["skills"], r["summary"], r["experience"]) if x)


def job_embed_text(j: dict) -> str:
    """Section-ordered text for embedding: title, requirements, then the description."""
    return " . ".join(x for x in (j["title"], j["requirements"], j["description"]) if x)
