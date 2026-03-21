
from endopigraph.utils import (
    ensure_dir,
    sha1_of_file,
    read_json,
    write_json,
    parse_percentile,
    pair_code,
    unpack_pair_code,
)


class TestEnsureDir:
    def test_creates(self, tmp_path):
        d = tmp_path / "a" / "b"
        r = ensure_dir(d)
        assert r.is_dir()

    def test_existing(self, tmp_path):
        r = ensure_dir(tmp_path)
        assert r.is_dir()


class TestSha1:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h1 = sha1_of_file(f)
        h2 = sha1_of_file(f)
        assert h1 == h2
        assert len(h1) == 40


class TestJson:
    def test_roundtrip(self, tmp_path):
        data = {"a": 1, "b": [2, 3]}
        p = tmp_path / "test.json"
        write_json(data, p)
        loaded = read_json(p)
        assert loaded == data


class TestParsePercentile:
    def test_valid(self):
        assert parse_percentile("percentile:95") == 95.0
        assert parse_percentile("Percentile:50") == 50.0

    def test_invalid(self):
        assert parse_percentile("otsu") is None
        assert parse_percentile("percentile") is None


class TestPairCode:
    def test_ordered(self):
        c1 = pair_code(1, 5)
        c2 = pair_code(5, 1)
        assert c1 == c2

    def test_roundtrip(self):
        c = pair_code(42, 99)
        a, b = unpack_pair_code(c)
        assert a == 42 and b == 99

    def test_roundtrip_reversed(self):
        c = pair_code(99, 42)
        a, b = unpack_pair_code(c)
        assert a == 42 and b == 99
