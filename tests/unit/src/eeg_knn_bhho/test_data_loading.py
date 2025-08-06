import pytest
from unittest.mock import patch, MagicMock
from eeg_knn_bhho.data_loading import load_data, preprocess_data, save_data

@pytest.fixture
def sample_data():
    return {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'label': [0, 1, 0]
    }

@pytest.fixture
def mock_file_system():
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        yield mock_open

def test_load_data_success(mock_file_system, sample_data):
    mock_file_system.return_value.__enter__.return_value.read.return_value = str(sample_data)
    data = load_data('dummy_path')
    assert data == sample_data

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_path')

def test_preprocess_data(sample_data):
    processed_data = preprocess_data(sample_data)
    assert 'processed_feature1' in processed_data
    assert 'processed_feature2' in processed_data

def test_preprocess_data_empty_input():
    with pytest.raises(ValueError):
        preprocess_data({})

def test_save_data_success(mock_file_system, sample_data):
    save_data('dummy_path', sample_data)
    mock_file_system.assert_called_once_with('dummy_path', 'w')
    mock_file_system().write.assert_called_once_with(str(sample_data))

def test_save_data_invalid_path():
    with pytest.raises(OSError):
        save_data('', sample_data)  # Testing with an invalid path

def test_load_data_invalid_format(mock_file_system):
    mock_file_system.return_value.__enter__.return_value.read.return_value = "invalid_format"
    with pytest.raises(ValueError):
        load_data('dummy_path')  # Assuming load_data raises ValueError for invalid format

def test_preprocess_data_edge_case():
    edge_case_data = {'feature1': [], 'feature2': [], 'label': []}
    processed_data = preprocess_data(edge_case_data)
    assert processed_data == {'processed_feature1': [], 'processed_feature2': []}  # Assuming it handles empty lists

def test_save_data_permission_error(mock_file_system, sample_data):
    mock_file_system.side_effect = OSError("Permission denied")
    with pytest.raises(OSError):
        save_data('protected_path', sample_data)  # Testing permission error scenario