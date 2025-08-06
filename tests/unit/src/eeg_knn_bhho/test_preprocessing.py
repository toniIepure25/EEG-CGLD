import pytest
from unittest.mock import patch
from eeg_knn_bhho.preprocessing import preprocess_data, normalize_data, split_data

@pytest.fixture
def sample_data():
    return [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

def test_preprocess_data(sample_data):
    # Test normal preprocessing
    processed_data = preprocess_data(sample_data)
    assert processed_data is not None
    assert len(processed_data) == len(sample_data)

def test_preprocess_data_empty_input():
    # Test preprocessing with empty input
    processed_data = preprocess_data([])
    assert processed_data == []

def test_normalize_data(sample_data):
    # Test normalization of data
    normalized_data = normalize_data(sample_data)
    assert normalized_data is not None
    assert all(len(row) == len(sample_data[0]) for row in normalized_data)

def test_normalize_data_with_zero_variance():
    # Test normalization with zero variance
    data = [[1, 1, 1], [1, 1, 1]]
    normalized_data = normalize_data(data)
    assert normalized_data == data  # Expect no change

def test_split_data(sample_data):
    # Test splitting data into training and test sets
    train_data, test_data = split_data(sample_data, test_size=0.33)
    assert len(train_data) + len(test_data) == len(sample_data)
    assert len(test_data) == 1  # Based on the test_size

def test_split_data_invalid_size(sample_data):
    # Test splitting with invalid test size
    with pytest.raises(ValueError):
        split_data(sample_data, test_size=1.5)

def test_split_data_empty_input():
    # Test splitting with empty input
    train_data, test_data = split_data([], test_size=0.33)
    assert train_data == []
    assert test_data == []