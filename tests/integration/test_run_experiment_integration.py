import pytest
from unittest.mock import patch, MagicMock
from run_experiment import run_experiment, ExperimentError

@pytest.fixture
def setup_experiment_data():
    # Setup realistic test data for the experiment
    return {
        "parameter1": 10,
        "parameter2": 20,
        "expected_result": 30
    }

@pytest.fixture
def mock_external_service():
    with patch('run_experiment.external_service') as mock_service:
        yield mock_service

def test_run_experiment_success(setup_experiment_data, mock_external_service):
    # Arrange
    mock_external_service.perform_calculation.return_value = setup_experiment_data["expected_result"]

    # Act
    result = run_experiment(setup_experiment_data["parameter1"], setup_experiment_data["parameter2"])

    # Assert
    assert result == setup_experiment_data["expected_result"]
    mock_external_service.perform_calculation.assert_called_once_with(10, 20)

def test_run_experiment_failure(mock_external_service):
    # Arrange
    mock_external_service.perform_calculation.side_effect = ExperimentError("Calculation failed")

    # Act & Assert
    with pytest.raises(ExperimentError, match="Calculation failed"):
        run_experiment(10, 20)

def test_run_experiment_edge_case_zero_parameters(mock_external_service):
    # Arrange
    mock_external_service.perform_calculation.return_value = 0

    # Act
    result = run_experiment(0, 0)

    # Assert
    assert result == 0
    mock_external_service.perform_calculation.assert_called_once_with(0, 0)

def test_run_experiment_edge_case_negative_parameters(mock_external_service):
    # Arrange
    mock_external_service.perform_calculation.return_value = -10

    # Act
    result = run_experiment(-5, -5)

    # Assert
    assert result == -10
    mock_external_service.perform_calculation.assert_called_once_with(-5, -5)

def test_run_experiment_invalid_input_type():
    # Act & Assert
    with pytest.raises(TypeError):
        run_experiment("invalid", 20)

def test_run_experiment_with_large_numbers(mock_external_service):
    # Arrange
    large_number1 = 10**6
    large_number2 = 10**6
    mock_external_service.perform_calculation.return_value = 2 * large_number1

    # Act
    result = run_experiment(large_number1, large_number2)

    # Assert
    assert result == 2 * large_number1
    mock_external_service.perform_calculation.assert_called_once_with(large_number1, large_number2)