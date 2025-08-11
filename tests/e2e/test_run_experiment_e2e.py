import pytest
from unittest.mock import patch, MagicMock
from run_experiment import run_experiment, ExperimentError

@pytest.fixture
def setup_experiment_data():
    # Setup mock data for the experiment
    return {
        "experiment_id": "exp123",
        "parameters": {"param1": 10, "param2": 20},
        "expected_result": {"success": True, "data": "Experiment completed successfully"}
    }

@pytest.fixture
def mock_external_service():
    with patch('run_experiment.external_service') as mock_service:
        yield mock_service

def test_run_experiment_success(setup_experiment_data, mock_external_service):
    # Arrange
    mock_external_service.perform_experiment.return_value = setup_experiment_data["expected_result"]
    
    # Act
    result = run_experiment(setup_experiment_data["experiment_id"], setup_experiment_data["parameters"])
    
    # Assert
    assert result == setup_experiment_data["expected_result"]
    mock_external_service.perform_experiment.assert_called_once_with(setup_experiment_data["experiment_id"], setup_experiment_data["parameters"])

def test_run_experiment_failure(setup_experiment_data, mock_external_service):
    # Arrange
    mock_external_service.perform_experiment.side_effect = ExperimentError("Experiment failed")
    
    # Act & Assert
    with pytest.raises(ExperimentError, match="Experiment failed"):
        run_experiment(setup_experiment_data["experiment_id"], setup_experiment_data["parameters"])

def test_run_experiment_invalid_parameters(setup_experiment_data, mock_external_service):
    # Arrange
    invalid_parameters = {"param1": -1, "param2": 20}  # Invalid parameter
    mock_external_service.perform_experiment.return_value = setup_experiment_data["expected_result"]
    
    # Act & Assert
    with pytest.raises(ValueError, match="Invalid parameters"):
        run_experiment(setup_experiment_data["experiment_id"], invalid_parameters)

def test_run_experiment_edge_case_empty_parameters(setup_experiment_data, mock_external_service):
    # Arrange
    empty_parameters = {}
    mock_external_service.perform_experiment.return_value = setup_experiment_data["expected_result"]
    
    # Act
    result = run_experiment(setup_experiment_data["experiment_id"], empty_parameters)
    
    # Assert
    assert result == setup_experiment_data["expected_result"]
    mock_external_service.perform_experiment.assert_called_once_with(setup_experiment_data["experiment_id"], empty_parameters)

def test_run_experiment_performance(setup_experiment_data, mock_external_service):
    # Arrange
    mock_external_service.perform_experiment.return_value = setup_experiment_data["expected_result"]
    
    # Act
    import time
    start_time = time.time()
    run_experiment(setup_experiment_data["experiment_id"], setup_experiment_data["parameters"])
    duration = time.time() - start_time
    
    # Assert
    assert duration < 2  # Ensure the experiment runs in under 2 seconds

def test_run_experiment_security_check(setup_experiment_data, mock_external_service):
    # Arrange
    mock_external_service.perform_experiment.return_value = setup_experiment_data["expected_result"]
    
    # Act
    result = run_experiment(setup_experiment_data["experiment_id"], setup_experiment_data["parameters"])
    
    # Assert
    assert "sensitive_data" not in result  # Ensure sensitive data is not returned in the result