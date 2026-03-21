
import pytest
import yaml
from endopigraph.config import load_config, _deep_merge


class TestDeepMerge:
    def test_flat(self):
        r = _deep_merge({"a": 1, "b": 2}, {"b": 3})
        assert r == {"a": 1, "b": 3}

    def test_nested(self):
        r = _deep_merge({"x": {"a": 1, "b": 2}}, {"x": {"b": 99}})
        assert r["x"]["a"] == 1
        assert r["x"]["b"] == 99

    def test_new_keys(self):
        r = _deep_merge({"a": 1}, {"b": 2})
        assert r == {"a": 1, "b": 2}

    def test_override_dict_with_scalar(self):
        r = _deep_merge({"a": {"x": 1}}, {"a": 42})
        assert r["a"] == 42


class TestLoadConfig:
    def test_valid(self, tmp_path):
        cfg = {"manifest_csv": "m.csv", "output_dir": "out/"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        c = load_config(p)
        assert c["manifest_csv"] == "m.csv"
        assert "segmentation" in c

    def test_missing_required(self, tmp_path):
        cfg = {"output_dir": "out/"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ValueError, match="manifest_csv"):
            load_config(p)

    def test_defaults_applied(self, tmp_path):
        cfg = {"manifest_csv": "m.csv", "output_dir": "out/"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        c = load_config(p)
        assert c["segmentation"]["method"] == "cellpose"
        assert c["graph"]["min_contact_px"] == 10

    def test_not_dict_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError):
            load_config(p)
