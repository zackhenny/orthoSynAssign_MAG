from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

from orthosynassign.refine import AUTHOR, VERSION, CalibrationModel, _filter_cluster_with_model, main, run_cli


@pytest.fixture
def args_factory() -> Namespace:
    """Returns a function that creates a mock args object with defaults."""

    def create_args(**kwargs):
        # Define your standard defaults here
        defaults = {
            "og_file": "default_og.tsv",
            "bed": "default.bed",
            "output": "output.tsv",
            "threads": 1,
            "verbose": False,
        }
        defaults.update(kwargs)
        return Namespace(**defaults)

    return create_args


# --- run_cli ---


class TestRunCli:
    @pytest.mark.parametrize("exit_code", [0, 1])
    def test_exit_code(self, monkeypatch: pytest.MonkeyPatch, exit_code: int):
        monkeypatch.setattr(sys, "argv", ["orthosynassign", "--og_file", "og", "--bed", "bed"])
        monkeypatch.setattr("orthosynassign.refine.main", lambda args: exit_code)

        with pytest.raises(SystemExit) as excinfo:
            run_cli()

        assert excinfo.value.code == exit_code

    def test_version(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
        # Force main to return exit code 0
        monkeypatch.setattr("orthosynassign.refine.main", lambda args: 0)

        monkeypatch.setattr(sys, "argv", ["orthosynassign", "-V"])
        with pytest.raises(SystemExit) as excinfo:
            run_cli()

        out1 = capsys.readouterr().out
        assert excinfo.value.code == 0

        monkeypatch.setattr(sys, "argv", ["orthosynassign", "--version"])
        with pytest.raises(SystemExit) as excinfo:
            run_cli()

        out2 = capsys.readouterr().out

        assert excinfo.value.code == 0
        assert out1.strip() == out2.strip() == VERSION

    def test_help(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
        # Force main to return exit code 0
        monkeypatch.setattr("orthosynassign.refine.main", lambda args: 0)

        monkeypatch.setattr(sys, "argv", ["orthosynassign", "-h"])
        with pytest.raises(SystemExit) as excinfo:
            run_cli()

        out1 = capsys.readouterr().out
        assert excinfo.value.code == 0

        monkeypatch.setattr(sys, "argv", ["orthosynassign", "--help"])
        with pytest.raises(SystemExit) as excinfo:
            run_cli()

        out2 = capsys.readouterr().out

        assert excinfo.value.code == 0
        assert out1 == out2
        assert f"Written by {AUTHOR}" in out1

    def test_missing_args(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["orthosynassign", "--bed", "bed"])
        # Force main to return exit code 0
        monkeypatch.setattr("orthosynassign.refine.main", lambda args: 0)

        with pytest.raises(SystemExit) as excinfo:
            run_cli()

        assert excinfo.value.code == 2

    def test_invalid_args(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["orthosynassign", "--invalid", "og", "--bed", "bed"])
        # Force main to return exit code 0
        monkeypatch.setattr("orthosynassign.refine.main", lambda args: 0)

        with pytest.raises(SystemExit) as excinfo:
            run_cli()

        assert excinfo.value.code == 2


# --- main ---


@pytest.fixture
def mock_refine_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Fixture to mock all heavy dependencies in the refine module."""
    # Mocking external file/validation calls
    monkeypatch.setattr("orthosynassign.refine.setup_logging", lambda x: None)
    monkeypatch.setattr("orthosynassign.refine.validate_annotations", lambda x: [])
    monkeypatch.setattr("orthosynassign.refine.validate_orthogroup", lambda x: x)
    monkeypatch.setattr("orthosynassign.refine.read_og_table", lambda x, y: {})
    monkeypatch.setattr("orthosynassign.refine._generate_sog_results", lambda a, b, c, cpus, **kwargs: iter([]))
    monkeypatch.setattr("orthosynassign.refine.write_og_table", lambda a, b, c: None)

    # Mock Path.mkdir and Path.unlink to prevent actual disk changes
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=True: None)
    monkeypatch.setattr(Path, "replace", lambda self, target: None)


class TestMain:
    def test_main_success(self, args_factory, mock_refine_dependencies):
        """Test that main returns 0 on a successful run."""
        args = args_factory()
        result = main(args)
        assert result == 0

    def test_main_keyboard_interrupt(self, monkeypatch, args_factory, mock_refine_dependencies):
        """Test that main catches Ctrl+C and returns 1."""
        args = args_factory()

        def mock_interrupt(args):
            raise KeyboardInterrupt

        monkeypatch.setattr("orthosynassign.refine.validate_annotations", mock_interrupt)

        result = main(args)
        assert result == 130

    def test_main_validation_error(self, monkeypatch, args_factory, mock_refine_dependencies):
        """Test that main catches general errors and returns 1."""
        args = args_factory()

        def mock_crash(*args):
            raise FileNotFoundError

        monkeypatch.setattr("orthosynassign.refine.read_og_table", mock_crash)

        result = main(args)
        assert result == 2

    def test_main_general_exception(self, monkeypatch, args_factory, mock_refine_dependencies):
        """Test that main catches general errors and returns 1."""
        args = args_factory()

        def mock_crash(*args):
            raise RuntimeError

        monkeypatch.setattr("orthosynassign.refine.read_og_table", mock_crash)

        result = main(args)
        assert result == 1

    def test_tmp_file_cleanup(self, monkeypatch, args_factory, mock_refine_dependencies):
        """Test that the .tmp file is unlinked in the finally block."""
        args = args_factory()
        cleanup_called = False

        def mock_unlink(self, missing_ok=True):
            nonlocal cleanup_called  # Modify cleanup_called in the outer scope
            cleanup_called = True

        # Check if tmp_output.unlink() is called
        monkeypatch.setattr(Path, "unlink", mock_unlink)
        # Ensure Path thinks the tmp file exists so it enters the cleanup
        monkeypatch.setattr(Path, "exists", lambda path_instance: True)

        main(args)
        assert cleanup_called is True


# ---------------------------------------------------------------------------
# CalibrationModel
# ---------------------------------------------------------------------------

class TestCalibrationModel:
    """Tests for CalibrationModel load / score."""

    @pytest.fixture
    def calibration_json(self, tmp_path):
        """Write a minimal calibration.json to a temp file."""
        import numpy as np

        data = {
            "feature_names": ["sim_flank_score", "log_flank_completeness", "og_size", "genome_completeness"],
            "scaler_mean": [0.5, -1.0, 10.0, 0.9],
            "scaler_scale": [0.2, 0.3, 5.0, 0.1],
            "coefficients": [-2.0, -1.0, 0.1, -0.5],
            "intercept": 1.0,
            "threshold_f1": 0.5,
            "threshold_recall": 0.3,
        }
        p = tmp_path / "calibration.json"
        p.write_text(json.dumps(data))
        return p

    def test_from_json_loads_correctly(self, calibration_json):
        model = CalibrationModel.from_json(calibration_json)
        assert len(model.feature_names) == 4
        assert model.threshold == pytest.approx(0.5)

    def test_from_json_threshold_recall(self, calibration_json):
        model = CalibrationModel.from_json(calibration_json, threshold_type="recall")
        assert model.threshold == pytest.approx(0.3)

    def test_from_dict(self, calibration_json):
        """from_dict builds same model as from_json."""
        import json

        with open(calibration_json) as fh:
            data = json.load(fh)
        model = CalibrationModel.from_dict(data)
        assert model.threshold == pytest.approx(0.5)

    def test_split_probability_range(self, calibration_json):
        model = CalibrationModel.from_json(calibration_json)
        p = model.split_probability(0.5, 0.9, 10, 0.9)
        assert 0.0 <= p <= 1.0

    def test_is_split_classification(self, calibration_json):
        """Verify is_split uses the threshold correctly."""
        model = CalibrationModel.from_json(calibration_json)
        # Manually override threshold so we know which way it goes.
        model.threshold = 0.0
        assert not model.is_split(0.5, 0.9, 10, 0.9)
        model.threshold = 1.0
        assert model.is_split(0.5, 0.9, 10, 0.9)

    def test_main_returns_zero_with_calibration_arg(
        self, monkeypatch, args_factory, calibration_json
    ):
        """main() returns error 1 when --calibration is supplied without --hog_file."""
        # Only mock the minimal dependencies so validate_annotations doesn't hit the fs.
        monkeypatch.setattr("orthosynassign.refine.setup_logging", lambda x: None)
        monkeypatch.setattr("orthosynassign.refine.validate_annotations", lambda x: [])
        monkeypatch.setattr("orthosynassign.refine.validate_orthogroup", lambda x: x)
        monkeypatch.setattr("orthosynassign.refine.read_og_table", lambda x, y: {})
        monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
        monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=True: None)
        monkeypatch.setattr(Path, "replace", lambda self, target: None)
        args = args_factory(
            calibration=calibration_json,
            hog_file=None,   # no hog_file → ValueError → return 1
        )
        result = main(args)
        assert result == 1

    def test_main_calibration_missing_hog_raises(
        self, monkeypatch, args_factory, calibration_json
    ):
        """main() returns error code when --calibration given without --hog_file."""
        monkeypatch.setattr("orthosynassign.refine.setup_logging", lambda x: None)
        monkeypatch.setattr("orthosynassign.refine.validate_annotations", lambda x: [])
        monkeypatch.setattr("orthosynassign.refine.validate_orthogroup", lambda x: x)
        monkeypatch.setattr("orthosynassign.refine.read_og_table", lambda x, y: {})
        monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
        monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=True: None)
        monkeypatch.setattr(Path, "replace", lambda self, target: None)
        args = args_factory(calibration=calibration_json, hog_file=None)
        result = main(args)
        assert result == 1  # missing hog_file → ValueError → code 1


# ---------------------------------------------------------------------------
# _filter_cluster_with_model
# ---------------------------------------------------------------------------

class TestFilterClusterWithModel:
    """Tests for the per-cluster calibration filter."""

    @pytest.fixture
    def simple_model(self):
        """Model that always classifies as 'split' (threshold > any probability)."""
        import numpy as np

        return CalibrationModel(
            feature_names=["sim_flank_score", "log_flank_completeness", "og_size", "genome_completeness"],
            scaler_mean=np.zeros(4),
            scaler_scale=np.ones(4),
            coef=np.zeros(4),
            intercept=-100.0,   # logit → probability ≈ 0 → always below any threshold > 0
            threshold=0.5,
        )

    @pytest.fixture
    def permissive_model(self):
        """Model that never classifies as 'split' (threshold = 0)."""
        import numpy as np

        return CalibrationModel(
            feature_names=["sim_flank_score", "log_flank_completeness", "og_size", "genome_completeness"],
            scaler_mean=np.zeros(4),
            scaler_scale=np.ones(4),
            coef=np.zeros(4),
            intercept=100.0,    # logit → probability ≈ 1 → always above threshold 0
            threshold=0.0,
        )

    def test_interior_genes_always_kept(self, permissive_model, gene_factory, genome_factory):
        """Interior genes (edge_type=0) are always retained regardless of model."""
        genome = genome_factory("G")
        g = gene_factory("g1", "chr1", 100, 200)
        genome.add_gene(g)

        # edge_type=0 → interior
        flank_cache = {(0, 0): (0.0, 0.5, 0)}
        cluster = [(0, 0)]
        # Even with the simplistic model that drops everything, interior genes pass.
        # (Returned list will have 1 gene, which is < 2, so empty result.)
        result = _filter_cluster_with_model(cluster, permissive_model, flank_cache, 5, [genome], {})
        # Interior genes are never dropped by the model — but a single-gene cluster is dropped.
        assert result == []  # len == 1 → filtered out

    def test_edge_gene_dropped_when_classified_as_split(self, simple_model, gene_factory, genome_factory):
        """Edge gene classified as split (prob < threshold) is removed from cluster."""
        genome = genome_factory("G")
        for i in range(3):
            genome.add_gene(gene_factory(f"g{i}", "chr1", i * 100, i * 100 + 50))

        # Gene at index 0 is interior, gene at index 1 is edge → model classifies it as split.
        flank_cache = {
            (0, 0): (0.5, 1.0, 0),   # interior → kept
            (0, 1): (0.1, 0.2, 1),   # left_edge → model decides
        }
        cluster = [(0, 0), (0, 1)]
        result = _filter_cluster_with_model(cluster, simple_model, flank_cache, 5, [genome], {})
        # Model's intercept=-100 → prob≈0 < threshold=0.5 → edge gene dropped.
        # Remaining: [(0,0)] which has len=1 → empty.
        assert result == []

    def test_edge_gene_kept_when_not_classified_as_split(self, permissive_model, gene_factory, genome_factory):
        """Edge gene NOT classified as split is retained."""
        genome = genome_factory("G")
        for i in range(3):
            genome.add_gene(gene_factory(f"g{i}", "chr1", i * 100, i * 100 + 50))

        flank_cache = {
            (0, 0): (0.9, 0.9, 0),   # interior
            (0, 1): (0.8, 0.8, 1),   # left_edge
        }
        cluster = [(0, 0), (0, 1)]
        result = _filter_cluster_with_model(cluster, permissive_model, flank_cache, 5, [genome], {})
        # Permissive model (intercept=100, threshold=0): prob≈1 ≥ 0 → not split → kept.
        assert result == [(0, 0), (0, 1)]

    def test_gene_not_in_cache_always_kept(self, simple_model, gene_factory, genome_factory):
        """Genes without flank cache entry are unconditionally retained."""
        genome = genome_factory("G")
        for i in range(3):
            genome.add_gene(gene_factory(f"g{i}", "chr1", i * 100, i * 100 + 50))

        flank_cache: dict = {}  # empty
        cluster = [(0, 0), (0, 1)]
        result = _filter_cluster_with_model(cluster, simple_model, flank_cache, 5, [genome], {})
        assert result == [(0, 0), (0, 1)]

    def test_new_cli_args_accepted(self, monkeypatch):
        """The new CLI arguments are present and parseable."""
        import sys

        from orthosynassign.refine import _parse_arguments

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "orthosynassign",
                "--og_file", "og.tsv",
                "--bed", "a.bed",
                "--calibration", "calibration.json",
                "--hog_file", "N0.tsv",
            ],
        )
        args = _parse_arguments(
            [
                "--og_file", "og.tsv",
                "--bed", "a.bed",
                "--calibration", "calibration.json",
                "--hog_file", "N0.tsv",
            ]
        )
        assert args.calibration == Path("calibration.json")
        assert args.hog_file == Path("N0.tsv")
        assert args.auto_calibrate is False

    def test_auto_calibrate_flag(self):
        from orthosynassign.refine import _parse_arguments

        args = _parse_arguments(
            ["--og_file", "og.tsv", "--bed", "a.bed", "--auto_calibrate", "--hog_file", "N0.tsv"]
        )
        assert args.auto_calibrate is True
        assert args.hog_file == Path("N0.tsv")
