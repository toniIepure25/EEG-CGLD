import pytest
from unittest.mock import patch, MagicMock
from src.eeg_knn_bhho.data_loading import load_data, preprocess_data, save_data

@pytest.fixture
def sample_data():
    return {
        'data': [[1, 2, 3], [4, 5, 6]],
        'labels': [0, 1]
    }

def test_load_data_success(sample_data):
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"data": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]}'
        data = load_data('dummy_path.json')
        assert data['data'] == sample_data['data']
        assert data['labels'] == sample_data['labels']

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_file.json')

def test_preprocess_data(sample_data):
    processed_data = preprocess_data(sample_data['data'])
    assert processed_data is not None
    assert len(processed_data) == len(sample_data['data'])
    assert all(isinstance(item, list) for item in processed_data)

def test_preprocess_data_empty_input():
    processed_data = preprocess_data([])
    assert processed_data == []

def test_save_data_success(sample_data):
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        save_data('dummy_path.json', sample_data)
        mock_open.assert_called_once_with('dummy_path.json', 'w')
        mock_open().write.assert_called_once()

def test_save_data_invalid_input():
    with pytest.raises(TypeError):
        save_data('dummy_path.json', None)