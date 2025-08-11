import pytest
from unittest.mock import patch, MagicMock
from src.eeg_knn_bhho.classification import Classifier

@pytest.fixture
def classifier():
    return Classifier()

def test_classifier_initialization(classifier):
    assert classifier is not None
    assert classifier.model is None  # Assuming model is initialized to None

def test_fit_method(classifier):
    mock_data = MagicMock()
    mock_labels = MagicMock()
    
    with patch('src.eeg_knn_bhho.classification.SomeModel') as MockModel:
        instance = MockModel.return_value
        classifier.fit(mock_data, mock_labels)
        MockModel.assert_called_once()
        instance.fit.assert_called_once_with(mock_data, mock_labels)

def test_predict_method(classifier):
    mock_data = MagicMock()
    classifier.model = MagicMock()
    classifier.model.predict.return_value = [0, 1, 0]

    predictions = classifier.predict(mock_data)

    classifier.model.predict.assert_called_once_with(mock_data)
    assert predictions == [0, 1, 0]

def test_predict_empty_data(classifier):
    mock_data = []
    classifier.model = MagicMock()
    
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        classifier.predict(mock_data)

def test_fit_invalid_data(classifier):
    mock_data = None
    mock_labels = None
    
    with pytest.raises(ValueError, match="Data and labels cannot be None"):
        classifier.fit(mock_data, mock_labels)

def test_model_training_with_invalid_labels(classifier):
    mock_data = MagicMock()
    mock_labels = [0, 1, 2]  # Assuming only binary classification is valid
    
    with pytest.raises(ValueError, match="Invalid labels provided"):
        classifier.fit(mock_data, mock_labels)

def test_model_prediction_with_unfitted_model(classifier):
    mock_data = MagicMock()
    
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        classifier.predict(mock_data)

def test_classifier_save_and_load(classifier):
    mock_data = MagicMock()
    mock_labels = MagicMock()
    classifier.fit(mock_data, mock_labels)

    with patch('src.eeg_knn_bhho.classification.joblib') as mock_joblib:
        classifier.save('model_path')
        mock_joblib.dump.assert_called_once_with(classifier.model, 'model_path')

    with patch('src.eeg_knn_bhho.classification.joblib') as mock_joblib:
        classifier.load('model_path')
        mock_joblib.load.assert_called_once_with('model_path')

def test_classifier_save_without_fit(classifier):
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        classifier.save('model_path')