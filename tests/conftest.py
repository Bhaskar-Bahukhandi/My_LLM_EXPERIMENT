"""Keep unchanged historical contracts in their required, verified replay context."""

# These entire modules still run through test_historical_authorization_replay.py.
# Running their __file__-relative ROOT at today's root checks superseded inputs.
# A default pytest run requires the replay test; missing/corrupt snapshots fail.
collect_ignore = [
    "test_authorization_capacity.py",
    "test_authorization_r3.py",
    "test_benchmark_fingerprints.py",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "historical_replay: mandatory verified immutable authorization replay"
    )
