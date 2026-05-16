from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mcis.utils.io import ensure_dir, load_yaml, save_dataframe, save_json, snapshot_config
from mcis.utils.logging import get_logger, log_output_path, log_row_count
from mcis.utils.seeds import set_global_seed


class TestIO:
    def test_ensure_dir_creates(self, tmp_path):
        d = tmp_path / "new_dir" / "nested"
        result = ensure_dir(d)
        assert result == d
        assert d.exists()

    def test_ensure_dir_existing(self, tmp_path):
        result = ensure_dir(tmp_path)
        assert result == tmp_path

    def test_load_yaml_valid(self, tmp_path):
        p = tmp_path / "test.yaml"
        with open(p, "w") as f:
            yaml.dump({"key": "value", "num": 42}, f)
        data = load_yaml(p)
        assert data == {"key": "value", "num": 42}

    def test_load_yaml_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_yaml("nonexistent.yaml")

    def test_save_json(self, tmp_path):
        obj = {"a": 1, "b": [2, 3]}
        path = tmp_path / "out.json"
        save_json(obj, path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == obj

    def test_save_json_creates_dir(self, tmp_path):
        path = tmp_path / "nested" / "out.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_save_dataframe_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.parquet"
        save_dataframe(df, path)
        assert path.exists()
        loaded = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_dataframe_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "test.csv"
        save_dataframe(df, path)
        assert path.exists()
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_dataframe_unsupported(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unsupported"):
            save_dataframe(df, tmp_path / "test.xlsx")

    def test_snapshot_config(self, tmp_path):
        cfg = {"project": {"name": "mcis", "seed": 42}}
        path = snapshot_config(cfg, tmp_path)
        assert path.exists()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == cfg

    def test_snapshot_config_creates_dir(self, tmp_path):
        path = snapshot_config({"a": 1}, tmp_path / "sub" / "dir")
        assert path.exists()


class TestLogging:
    def test_get_logger_returns_logger(self):
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == 20

    def test_get_logger_with_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = get_logger("file_logger", log_file=log_file)
        logger.info("hello")
        assert log_file.exists()
        content = log_file.read_text()
        assert "hello" in content

    def test_get_logger_reuses_handlers(self):
        logger1 = get_logger("reuse_test")
        logger2 = get_logger("reuse_test")
        assert logger1 is logger2

    def test_log_row_count(self, tmp_path, caplog):
        import logging
        log_file = tmp_path / "count.log"
        logger = get_logger("count_test", log_file=log_file, level=logging.DEBUG)
        log_row_count(logger, "clean", 100, 85)
        content = log_file.read_text(encoding="utf-8")
        assert "clean" in content
        assert "100" in content
        assert "85" in content

    def test_log_output_path(self, tmp_path, caplog):
        log_file = tmp_path / "path.log"
        logger = get_logger("path_test", log_file=log_file)
        log_output_path(logger, tmp_path / "output.parquet")
        content = log_file.read_text()
        assert "output.parquet" in content


class TestSeeds:
    def test_set_global_seed_runs(self):
        set_global_seed(42)

    def test_set_global_seed_reproducible(self):
        set_global_seed(123)
        import random
        a = random.randint(0, 1000)
        set_global_seed(123)
        b = random.randint(0, 1000)
        assert a == b
