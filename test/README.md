# Tests

Run everything with the standard library, from the repository root:

```bash
python -m unittest discover -s test
python -m unittest discover -s test -v
python -m unittest discover -s test -p "test_nn_pair_variability.py"
```

A missing dependency (torch, `ab.nn`) or missing data (no LEMUR database)
reports as a skip, not an error. `pytest test/` keeps working.

## Writing a test

A `unittest.TestCase` is collected as it is. A file in the pytest idiom —
plain `test_*` functions, `tmp_path` / `monkeypatch`, `parametrize`, `fixture` —
needs two lines, because `unittest` collects neither functions nor fixtures:

```python
from _unittest_bridge import load_tests_from, pytest   # instead of: import pytest
...
load_tests = load_tests_from(globals())                # at the end of the file
```

For an import that may fail:

```python
try:
    import torch
except ImportError as exc:
    from _unittest_bridge import skip_module
    skip_module(f"needs torch: {exc}")
```

## Scope of the LEMUR tests

`test.py` and `test_nn_pair_variability.py` query the database:

| variable | default | meaning |
| --- | --- | --- |
| `LEMUR_TASK` | `img-classification` | task filter |
| `LEMUR_DATASET` | `cifar-10` | dataset filter |
| `LEMUR_METRIC` | `acc` | metric filter |
| `LEMUR_PAIR_CALLS` | `5` | how often a pair request is repeated |
| `LEMUR_PAIR_ROWS` | `200` | rows per pair request |

`NNGPT_TEST_TOKENIZER=<model>` enables the `test_nngen` case that needs a
tokenizer from the hub; without it nothing is downloaded.
