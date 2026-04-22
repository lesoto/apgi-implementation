"""Tests for validation/empirical_validation.py dataset-driven validation."""

import tempfile
from unittest.mock import MagicMock

import numpy as np

from validation.empirical_validation import (
    DatasetConfig,
    EmpiricalDataLoader,
    NeuralValidator,
    BehavioralValidator,
    CrossValidationRunner,
    create_synthetic_validation_dataset,
)


class TestDatasetConfig:
    """Test DatasetConfig dataclass."""

    def test_dataset_config_creation(self):
        """Test DatasetConfig creation."""
        config = DatasetConfig(
            name="test_dataset",
            data_type="eeg",
            fs=100.0,
            duration=10.0,
            n_trials=50,
            condition_labels=["A", "B"],
        )
        assert config.name == "test_dataset"
        assert config.data_type == "eeg"
        assert config.fs == 100.0
        assert config.duration == 10.0
        assert config.n_trials == 50
        assert config.condition_labels == ["A", "B"]


class TestEmpiricalDataLoader:
    """Test empirical data loading."""

    def test_initialization(self):
        """Test EmpiricalDataLoader initialization."""
        config = DatasetConfig(name="test", data_type="simulation", fs=100.0)
        loader = EmpiricalDataLoader(config)
        assert loader.config == config
        assert loader.data == {}
        assert loader.metadata == {}
        assert loader._behavioral_data == {}

    def test_load_eeg_dataset_fallback_numpy(self):
        """Test EEG loading with NumPy fallback (no MNE)."""
        from unittest.mock import patch

        config = DatasetConfig(name="test", data_type="eeg", fs=100.0)
        loader = EmpiricalDataLoader(config)

        # Create a temporary .npy file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".npy") as f:
            filepath = f.name

        try:
            # Create test data
            test_data = np.random.randn(2, 1000)
            np.save(filepath, test_data)

            # Mock MNE import to force fallback
            with patch.dict("sys.modules", {"mne": None}):
                result = loader.load_eeg_dataset(filepath)
            assert "signals" in result
            assert "fs" in result
            assert "channels" in result
            assert result["signals"].shape == (2, 1000)
        finally:
            import os

            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_behavioral_dataset_csv(self):
        """Test behavioral dataset loading from CSV."""
        config = DatasetConfig(name="test", data_type="behavior", fs=100.0)
        loader = EmpiricalDataLoader(config)

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            filepath = f.name
            f.write("rt,accuracy,condition\n")
            f.write("0.5,1.0,A\n")
            f.write("0.6,0.0,B\n")
            f.write("0.4,1.0,A\n")

        try:
            result = loader.load_behavioral_dataset(filepath)
            assert "rt" in result
            assert "accuracy" in result
            assert "n_trials" in result
            assert result["n_trials"] == 3
            assert len(result["rt"]) == 3
        finally:
            import os

            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_behavioral_dataset_json(self):
        """Test behavioral dataset loading from JSON."""
        config = DatasetConfig(name="test", data_type="behavior", fs=100.0)
        loader = EmpiricalDataLoader(config)

        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name
            f.write('{"rt":[0.5,0.6,0.4],"accuracy":[1.0,0.0,1.0]}')

        try:
            result = loader.load_behavioral_dataset(filepath)
            assert "rt" in result
            assert "accuracy" in result
            assert len(result["rt"]) == 3
        finally:
            import os

            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_behavioral_dataset_with_conditions(self):
        """Test behavioral dataset loading with conditions."""
        config = DatasetConfig(name="test", data_type="behavior", fs=100.0)
        loader = EmpiricalDataLoader(config)

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            filepath = f.name
            f.write("rt,accuracy,condition\n")
            f.write("0.5,1.0,A\n")
            f.write("0.6,0.0,B\n")
            f.write("0.4,1.0,A\n")

        try:
            result = loader.load_behavioral_dataset(
                filepath, condition_column="condition"
            )
            assert "conditions" in result
            assert "by_condition" in result
            assert "A" in result["by_condition"]
            assert "B" in result["by_condition"]
        finally:
            import os

            if os.path.exists(filepath):
                os.remove(filepath)

    def test_get_segment_no_data_loaded(self):
        """Test get_segment raises error when no data loaded."""
        config = DatasetConfig(name="test", data_type="simulation", fs=100.0)
        loader = EmpiricalDataLoader(config)

        try:
            loader.get_segment(0.0, 1.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No data loaded" in str(e)

    def test_get_segment_with_data(self):
        """Test get_segment extracts correct segment."""
        config = DatasetConfig(name="test", data_type="simulation", fs=100.0)
        loader = EmpiricalDataLoader(config)

        # Load test data
        loader.data = {
            "signals": np.random.randn(2, 1000),
            "fs": 100.0,
        }

        segment = loader.get_segment(0.0, 1.0, channel_idx=0)
        assert segment.shape == (100,)  # 1 second at 100 Hz


class TestNeuralValidator:
    """Test neural data validation."""

    def test_initialization(self):
        """Test NeuralValidator initialization."""
        validator = NeuralValidator(fs=100.0)
        assert validator.fs == 100.0

    def test_extract_gamma_power_welch(self):
        """Test gamma power extraction with Welch method."""
        validator = NeuralValidator(fs=100.0)
        signal = np.random.randn(500)
        power = validator.extract_gamma_power(signal, method="welch")
        assert isinstance(power, float)
        assert power >= 0.0

    def test_extract_gamma_power_fft(self):
        """Test gamma power extraction with FFT method."""
        validator = NeuralValidator(fs=100.0)
        signal = np.random.randn(500)
        power = validator.extract_gamma_power(signal, method="fft")
        assert isinstance(power, float)
        assert power >= 0.0

    def test_extract_gamma_power_custom_freq_range(self):
        """Test gamma power with custom frequency range."""
        validator = NeuralValidator(fs=100.0)
        signal = np.random.randn(500)
        power = validator.extract_gamma_power(signal, freq_range=(20, 50))
        assert isinstance(power, float)

    def test_compute_erp_list_epochs(self):
        """Test ERP computation with list of epochs."""
        validator = NeuralValidator(fs=100.0)
        epochs = [np.random.randn(100) for _ in range(10)]
        result = validator.compute_erp(epochs)
        assert "erp" in result
        assert "times" in result
        assert "n_epochs" in result
        assert "p300_amplitude" in result
        assert "n200_amplitude" in result
        assert result["n_epochs"] == 10

    def test_compute_erp_array_epochs(self):
        """Test ERP computation with array of epochs."""
        validator = NeuralValidator(fs=100.0)
        epochs = np.random.randn(10, 100)
        result = validator.compute_erp(epochs)
        assert "erp" in result
        assert result["n_epochs"] == 10

    def test_compute_erp_custom_windows(self):
        """Test ERP with custom time windows."""
        validator = NeuralValidator(fs=100.0)
        epochs = np.random.randn(10, 100)
        result = validator.compute_erp(
            epochs,
            baseline=(-0.1, 0.0),
            time_window=(-0.1, 0.5),
        )
        assert "erp" in result

    def test_validate_against_apgi(self):
        """Test validation against APGI results."""
        validator = NeuralValidator(fs=100.0)
        neural_data = np.random.randn(500)
        apgi_result = {
            "S": 1.0,
            "S_history": [0.8, 0.9, 1.0, 1.1, 1.2],
        }
        result = validator.validate_against_apgi(apgi_result, neural_data)
        assert "neural_gamma_power" in result
        assert "apgi_s_proxy" in result
        assert "prediction_error" in result


class TestBehavioralValidator:
    """Test behavioral data validation."""

    def test_initialization(self):
        """Test BehavioralValidator initialization."""
        validator = BehavioralValidator()
        assert validator.data == {}

    def test_compute_signal_detection_metrics(self):
        """Test signal detection theory metrics."""
        validator = BehavioralValidator()
        result = validator.compute_signal_detection_metrics(
            hits=50,
            misses=10,
            false_alarms=20,
            correct_rejections=120,
        )
        assert "d_prime" in result
        assert "criterion" in result
        assert "beta" in result
        assert "hit_rate" in result
        assert "fa_rate" in result
        assert isinstance(result["d_prime"], float)

    def test_compute_signal_detection_metrics_edge_cases(self):
        """Test SDT with edge cases (all hits, no FAs)."""
        validator = BehavioralValidator()
        result = validator.compute_signal_detection_metrics(
            hits=100,
            misses=0,
            false_alarms=0,
            correct_rejections=100,
        )
        # Should handle edge cases with correction
        assert "d_prime" in result

    def test_analyze_rt_distribution(self):
        """Test RT distribution analysis."""
        validator = BehavioralValidator()
        rts = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        result = validator.analyze_rt_distribution(rts)
        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert "cv" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert "n" in result
        assert result["n"] == 6

    def test_analyze_rt_distribution_outlier_removal(self):
        """Test RT distribution with outlier removal."""
        validator = BehavioralValidator()
        rts = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 10.0])  # 10.0 is a clear outlier
        result = validator.analyze_rt_distribution(
            rts, remove_outliers=True, outlier_sd=2.0
        )
        assert result["n"] < 6  # Should have removed outlier

    def test_compute_skewness(self):
        """Test skewness computation."""
        skewness = BehavioralValidator._compute_skewness(np.array([1, 2, 3, 4, 5]))
        assert isinstance(skewness, float)

    def test_compute_kurtosis(self):
        """Test kurtosis computation."""
        kurtosis = BehavioralValidator._compute_kurtosis(np.array([1, 2, 3, 4, 5]))
        assert isinstance(kurtosis, float)

    def test_validate_against_apgi(self):
        """Test validation against APGI results."""
        validator = BehavioralValidator()
        apgi_result = {"theta": 0.7}
        behavioral_data = {
            "rt_std": 0.15,
            "criterion": 0.5,
        }
        result = validator.validate_against_apgi(apgi_result, behavioral_data)
        assert "apgi_theta" in result
        assert "behavioral_rt_std" in result
        assert "behavioral_criterion" in result
        assert "correlation_proxy" in result


class TestCrossValidationRunner:
    """Test cross-validation runner."""

    def test_initialization(self):
        """Test CrossValidationRunner initialization."""
        runner = CrossValidationRunner(n_folds=5)
        assert runner.n_folds == 5
        assert runner.results == []

    def test_split_data_basic(self):
        """Test basic data splitting."""
        runner = CrossValidationRunner(n_folds=5)
        data = np.random.randn(100)
        splits = runner.split_data(data)
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == 100

    def test_split_data_with_labels(self):
        """Test data splitting with labels (placeholder)."""
        runner = CrossValidationRunner(n_folds=3)
        data = np.random.randn(30)
        labels = np.array([0, 1] * 15)
        splits = runner.split_data(data, labels)
        assert len(splits) == 3

    def test_run_validation(self):
        """Test cross-validation execution."""
        runner = CrossValidationRunner(n_folds=3)
        data = np.random.randn(30)

        # Mock APGI pipeline
        mock_pipeline = MagicMock()

        def metric_fn(test_data, pipeline):
            return np.mean(test_data)

        result = runner.run_validation(data, mock_pipeline, metric_fn)
        assert "fold_scores" in result
        assert "mean_score" in result
        assert "std_score" in result
        assert "n_folds" in result
        assert len(result["fold_scores"]) == 3


class TestCreateSyntheticValidationDataset:
    """Test synthetic validation dataset creation."""

    def test_create_dataset_basic(self):
        """Test basic synthetic dataset creation."""
        dataset = create_synthetic_validation_dataset(
            n_samples=10,
            duration=5.0,
            fs=100.0,
        )
        assert "data" in dataset
        assert "fs" in dataset
        assert "n_samples" in dataset
        assert "duration" in dataset
        assert dataset["data"].shape == (10, 500)  # 10 samples, 5s at 100Hz

    def test_create_dataset_with_ground_truth(self):
        """Test dataset creation with ground truth."""
        dataset = create_synthetic_validation_dataset(
            n_samples=5,
            duration=2.0,
            fs=100.0,
            with_ground_truth=True,
        )
        assert "ground_truth" in dataset
        assert "taus" in dataset["ground_truth"]
        assert "sigma2s" in dataset["ground_truth"]
        assert "beta" in dataset["ground_truth"]

    def test_create_dataset_without_ground_truth(self):
        """Test dataset creation without ground truth."""
        dataset = create_synthetic_validation_dataset(
            n_samples=5,
            duration=2.0,
            fs=100.0,
            with_ground_truth=False,
        )
        assert "ground_truth" not in dataset

    def test_create_dataset_custom_parameters(self):
        """Test dataset with custom parameters."""
        dataset = create_synthetic_validation_dataset(
            n_samples=20,
            duration=10.0,
            fs=200.0,
        )
        assert dataset["n_samples"] == 20
        assert dataset["duration"] == 10.0
        assert dataset["fs"] == 200.0
        assert dataset["data"].shape == (20, 2000)
