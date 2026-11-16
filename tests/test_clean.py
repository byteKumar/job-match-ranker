from jobmatch.clean import resume_embed_text, resume_sections, structure_job, structure_resume

HTML = ('<div class="section firstsection" id="SECTION_NAME1" style="x"><span> DATA ANALYST</span></div>'
        '<div class="section" id="SECTION_SUMM2" style="x"><div class="heading"><div class="sectiontitle" id="S">'
        ' Summary</div></div><p>Analyst at Company Name City , State</p></div>'
        '<div class="section" id="SECTION_SKLL3" style="x"><ul><li>SQL</li><li>Python</li></ul></div>')


def test_resume_sections_strip_html_and_titles():
    secs = resume_sections(HTML)
    assert secs["NAME"] == "DATA ANALYST"
    assert "Summary" not in secs["SUMM"]          # section heading removed
    assert "Company Name" not in secs["SUMM"]     # anonymization placeholder removed
    assert "style=" not in secs["SKLL"] and "SQL ; Python" in secs["SKLL"]


def test_embed_text_puts_title_and_skills_first():
    r = structure_resume(HTML, "DATA ANALYST Summary ...")
    assert resume_embed_text(r).startswith("DATA ANALYST . SQL ; Python")


def test_job_requirements_extracted():
    j = structure_job("Cook", "We are a diner. Requirements: 2 years line cook experience.")
    assert j["requirements"].startswith("Requirements")
    assert j["full"].startswith("Cook . ")


def test_uuid_section_ids_parse():
    # ~7% of resumes use UUID section ids instead of numeric ones.
    html = ('<div class="section firstsection" id="SECTION_NAMEf84d1218-3d6b" style="x"><span> CONSULTANT</span></div>'
            '<div class="section" id="SECTION_EXPR24ae643c-d0d9" style="x"><span>Managed GIS data</span></div>')
    secs = resume_sections(html)
    assert secs == {"NAME": "CONSULTANT", "EXPR": "Managed GIS data"}
