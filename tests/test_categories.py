import pytest

from jobmatch.categories import CATEGORIES, RELATED, grade, title_to_category


@pytest.mark.parametrize("title,expected", [
    ("Senior Software Engineer", "INFORMATION-TECHNOLOGY"),
    ("Data Engineer", "INFORMATION-TECHNOLOGY"),        # must beat the generic engineer rule
    ("Mechanical Engineer", "ENGINEERING"),
    ("Staff Accountant", "ACCOUNTANT"),
    ("Customer Service Representative", "BPO"),
    ("Inside Sales Representative", "SALES"),
    ("Business Development Representative", "BUSINESS-DEVELOPMENT"),
    ("Med/Surg RN", "HEALTHCARE"),
    ("Associate Attorney", "ADVOCATE"),
    ("Sous Chef", "CHEF"),
    ("CDL A City P&D Drivers, $27.61 hr", None),         # pay rate must not read as "HR"
    ("Nurse Anesthetist - Make $160/hr", "HEALTHCARE"),
    ("Communications/Radar Engineer", "ENGINEERING"),    # not public relations
    ("Server Farm Engineer", "ENGINEERING"),             # not agriculture
    ("Warehouse Associate", None),
])
def test_title_rules(title, expected):
    assert title_to_category(title) == expected


def test_grades():
    assert grade("ACCOUNTANT", "ACCOUNTANT") == 2
    assert grade("ACCOUNTANT", "FINANCE") == 1
    assert grade("FINANCE", "ACCOUNTANT") == 1      # related is symmetric
    assert grade("CHEF", "AVIATION") == 0


def test_related_is_symmetric_and_known():
    for a, bs in RELATED.items():
        assert a in CATEGORIES
        for b in bs:
            assert a in RELATED[b]
