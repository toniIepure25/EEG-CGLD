import pytest
from unittest.mock import patch
from eeg_knn_bhho.ica_preprocessing import preprocess_data, apply_ica

@pytest.fixture
def sample_data():
    # Sample data for testing
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

def test_preprocess_data_valid(sample_data):
    """Test preprocessing of valid data."""
    processed_data = preprocess_data(sample_data)
    assert processed_data is not None
    assert len(processed_data) == len(sample_data)

def test_preprocess_data_empty():
    """Test preprocessing with empty data."""
    processed_data = preprocess_data([])
    assert processed_data == []

def test_preprocess_data_invalid_type():
    """Test preprocessing with invalid data type."""
    with pytest.raises(TypeError):
        preprocess_data("invalid_data")

@patch('eeg_knn_bhho.ica_preprocessing.ica_algorithm')
def test_apply_ica_success(mock_ica_algorithm, sample_data):
    """Test applying ICA on valid data."""
    mock_ica_algorithm.return_value = [[0, 1], [1, 0]]
    ica_result = apply_ica(sample_data)
    assert ica_result is not None
    assert len(ica_result) == len(sample_data)

@patch('eeg_knn_bhho.ica_preprocessing.ica_algorithm')
def test_apply_ica_empty_data(mock_ica_algorithm):
    """Test applying ICA on empty data."""
    mock_ica_algorithm.return_value = []
    ica_result = apply_ica([])
    assert ica_result == []

@patch('eeg_knn_bhho.ica_preprocessing.ica_algorithm')
def test_apply_ica_invalid_data(mock_ica_algorithm):
    """Test applying ICA with invalid data type."""
    with pytest.raises(TypeError):
        apply_ica("invalid_data")

def test_apply_ica_edge_case():
    """Test applying ICA on edge case data."""
    edge_case_data = [[0, 0, 0], [0, 0, 0]]
    result = apply_ica(edge_case_data)
    assert result is not None
    assert all(all(value == 0 for value in row) for row in result)  # Assuming ICA would return zeros for zero input

@pytest.mark.parametrize("input_data, expected_output", [
    ([[1, 2], [3, 4]], [[0.5, 0.5], [0.5, 0.5]]),  # Example expected output
    ([[5, 6], [7, 8]], [[0.5, 0.5], [0.5, 0.5]])   # Example expected output
])
def test_preprocess_data_parametrized(input_data, expected_output):
    """Test preprocessing with parameterized input."""
    processed_data = preprocess_data(input_data)
    assert processed_data == expected_output