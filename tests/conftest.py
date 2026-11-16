import os

# Some tests skip locally when the data (scripts/prepare_data.py) or the bi-encoder model has not
# been downloaded. CI downloads both and sets JOBMATCH_STRICT_TESTS=1, so a skip there fails the run
# instead of passing quietly.


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if os.environ.get("JOBMATCH_STRICT_TESTS") and skipped:
        print(f"\nJOBMATCH_STRICT_TESTS: {len(skipped)} test(s) skipped, failing the run")
        session.exitstatus = 1
