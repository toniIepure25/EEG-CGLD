```python
# test_helper.py

import pytest
from unittest.mock import MagicMock

# Common setup and teardown functions
@pytest.fixture(scope='module', autouse=True)
def setup_module():
    # Setup code here
    yield
    # Teardown code here

@pytest.fixture(scope='function', autouse=True)
def setup_function():
    # Setup code for each test
    yield
    # Teardown code for each test

# Mock objects and test data
@pytest.fixture
def mock_data():
    return {
        'key1': 'value1',
        'key2': 'value2',
        'key3': MagicMock(return_value='mocked_value')
    }

@pytest.fixture
def sample_test_data():
    return [
        {'input': 1, 'expected': 2},
        {'input': 2, 'expected': 4},
        {'input': 3, 'expected': 6}
    ]

# Utility functions for testing
def assert_equal(actual, expected):
    assert actual == expected, f'Expected {expected}, but got {actual}'

def assert_in(item, collection):
    assert item in collection, f'{item} not found in {collection}'

# Configuration for python testing
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "fast: mark test as fast")
```
