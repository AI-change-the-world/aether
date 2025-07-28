import pytest

import aether


@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    aether.init_db()
