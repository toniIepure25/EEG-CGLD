import pytest
from unittest.mock import patch
from src.eeg_knn_bhho.preprocessing import preprocess_data, normalize_data, handle_missing_values

class TestPreprocessing:

    @pytest.fixture
    def sample_data(self):
        return [
            [1.0, 2.0, None],
            [4.0, 5.0, 6.0],
            [7.0, None, 9.0],
            [None, 11.0, 12.0]
        ]

    def test_handle_missing_values(self, sample_data):
        expected_output = [
            [1.0, 2.0, 9.0],
            [4.0, 5.0, 6.0],
            [7.0, 9.0, 9.0],
            [9.0, 11.0, 12.0]
        ]
        result = handle_missing_values(sample_data)
        assert result == expected_output

    def test_normalize_data(self):
        data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        expected_output = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0]
        ]
        result = normalize_data(data)
        assert result == expected_output

    @patch('src.eeg_knn_bhho.preprocessing.normalize_data')
    def test_preprocess_data(self, mock_normalize):
        mock_normalize.return_value = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0]
        ]
        data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        expected_output = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0]
        ]
        result = preprocess_data(data)
        assert result == expected_output
        mock_normalize.assert_called_once_with(data)

    def test_handle_missing_values_empty(self):
        result = handle_missing_values([])
        assert result == []

    def test_normalize_data_single_value(self):
        data = [[5.0]]
        expected_output = [[0.0]]
        result = normalize_data(data)
        assert result == expected_output

    def test_handle_missing_values_no_missing(self):
        data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        result = handle_missing_values(data)
        assert result == data

    def test_normalize_data_negative_values(self):
        data = [
            [-1.0, -2.0, -3.0],
            [-4.0, -5.0, -6.0],
            [-7.0, -8.0, -9.0]
        ]
        expected_output = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0]
        ]
        result = normalize_data(data)
        assert result == expected_output