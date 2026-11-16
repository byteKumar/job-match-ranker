"""Map job titles onto the resume dataset's 24 categories, and define graded relevance.

The resume dataset labels each resume with one of 24 occupational categories. Job postings
have no such label, so we derive one from the job title with ordered keyword rules. The
first rule that matches wins, so specific rules come before generic ones (for example
"data engineer" must hit INFORMATION-TECHNOLOGY before the generic "engineer" rule).

These are *weak* labels: cheap and consistent, but wrong some of the time. The hand-check
in scripts/handcheck.py measures how often.
"""

from __future__ import annotations

import re

CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE", "AVIATION",
    "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION", "CONSULTANT",
    "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]

# Ordered (category, regex) rules applied to the lowercased job title.
_RULES: list[tuple[str, str]] = [
    ("AVIATION", r"\b(pilot|aviation|aircraft|airframe|avionics|a&p|flight (attendant|instructor|dispatcher)|airport|aircrew)\b"),
    ("AUTOMOBILE", r"\b(automotive|auto (technician|mechanic|body)|diesel (technician|mechanic)|collision|vehicle technician)\b"),
    ("CHEF", r"\b(chef|cook|sous|culinary|kitchen|pastry)\b"),
    ("BUSINESS-DEVELOPMENT", r"\b(business development|bdr|growth manager|strategic alliances?)\b"),
    ("BPO", r"\b(call center|contact center|customer service|customer support|customer care|csr|bpo)\b"),
    ("SALES", r"\b(sales|salesperson|account executive|account manager|retail associate|sales associate)\b"),
    ("HR", r"\b(human resources|hr|recruiter|recruiting|talent acquisition|compensation|benefits|payroll|people operations)\b"),
    ("ADVOCATE", r"\b(attorney|lawyer|paralegal|legal|counsel|litigation)\b"),
    ("ACCOUNTANT", r"\b(accountant|accounting|bookkeeper|accounts (payable|receivable)|auditor|audit|tax|taxation|controller|cpa)\b"),
    ("BANKING", r"\b(bank|banker|banking|teller|loan|mortgage|underwriter|credit analyst)\b"),
    ("FINANCE", r"\b(financial|finance|fp&a|investment|treasury|cfo|wealth|portfolio)\b"),
    ("TEACHER", r"\b(teacher|teaching|tutor|educator|professor|paraprofessional|instructor)\b"),
    ("FITNESS", r"\b(fitness|personal trainer|yoga|pilates|gym|strength coach)\b"),
    ("HEALTHCARE", r"\b(nurse|nursing|rn|lpn|cna|physician|medical|clinical|therapist|pharmacy|pharmacist|dental|hygienist|patient|health|healthcare|technologist|surgical|caregiver|hospital|sonographer|phlebotomist|radiology)\b"),
    ("PUBLIC-RELATIONS", r"^(?!.*\bengineer).*\b(public relations|pr|communications|media relations|publicist)\b"),
    ("DIGITAL-MEDIA", r"\b(social media|digital|seo|content|video|multimedia|copywriter|writer|editor|producer|marketing)\b"),
    ("APPAREL", r"\b(apparel|fashion|garment|textile|sewing|tailor)\b"),
    ("DESIGNER", r"\b(designer|graphic|ux|ui|interior design)\b"),
    ("ARTS", r"\b(artist|art|arts|music|musician|photographer|museum|theater|theatre|animator|illustrator)\b"),
    ("AGRICULTURE", r"\b(agriculture|agricultural|(?<!server )farm|farmer|agronom\w*|crop|greenhouse|horticultur\w*|livestock|ranch hand)\b"),
    ("CONSTRUCTION", r"\b(construction|superintendent|carpenter|electrician|plumber|foreman|estimator|hvac|laborer|mason|roofer)\b"),
    ("INFORMATION-TECHNOLOGY", r"\b(software|developer|programmer|it|network|system administrator|systems administrator|devops|cloud|data (engineer|scientist|analyst)|cyber\w*|security analyst|help desk|desktop support|database|dba|sre|web|full stack|java|python|sap|servicenow|salesforce|qa)\b"),
    ("CONSULTANT", r"\b(consultant|consulting|advisory)\b"),
    ("ENGINEERING", r"\b(engineer|engineering)\b"),
]
_COMPILED = [(cat, re.compile(rx)) for cat, rx in _RULES]
# Pay rates in titles ("Make $160/hr", "$27.61 hr") would otherwise match the HR rule.
_PAY = re.compile(r"\$[\d,.]+k?\s*(/\s*)?(hr|hour|wk|week|yr|year)?\b|/\s*(hr|hour)\b")

# Symmetric "related field" pairs. A resume in one and a job in the other gets grade 1.
_RELATED_PAIRS = [
    ("ACCOUNTANT", "FINANCE"), ("ACCOUNTANT", "BANKING"), ("FINANCE", "BANKING"),
    ("FINANCE", "CONSULTANT"), ("INFORMATION-TECHNOLOGY", "ENGINEERING"),
    ("SALES", "BUSINESS-DEVELOPMENT"), ("SALES", "BPO"), ("BUSINESS-DEVELOPMENT", "CONSULTANT"),
    ("DIGITAL-MEDIA", "ARTS"), ("DIGITAL-MEDIA", "DESIGNER"), ("DIGITAL-MEDIA", "PUBLIC-RELATIONS"),
    ("DESIGNER", "ARTS"), ("DESIGNER", "APPAREL"), ("HEALTHCARE", "FITNESS"),
    ("CONSTRUCTION", "ENGINEERING"), ("AVIATION", "ENGINEERING"), ("AUTOMOBILE", "ENGINEERING"),
    ("HR", "CONSULTANT"),
]
RELATED: dict[str, set[str]] = {c: set() for c in CATEGORIES}
for a, b in _RELATED_PAIRS:
    RELATED[a].add(b)
    RELATED[b].add(a)


def title_to_category(title: str) -> str | None:
    """Return the first category whose rule matches the title, or None if none match."""
    t = _PAY.sub(" ", (title or "").lower())
    for cat, rx in _COMPILED:
        if rx.search(t):
            return cat
    return None


def grade(resume_cat: str, job_cat: str) -> int:
    """Weak relevance grade: 2 same category, 1 related category, 0 otherwise."""
    if resume_cat == job_cat:
        return 2
    if job_cat in RELATED.get(resume_cat, ()):
        return 1
    return 0
