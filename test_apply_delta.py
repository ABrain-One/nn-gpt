"""Regression test for ab/gpt/util/nn/DeltaUtil.apply_delta: a hunk whose context/removed lines do not
occur in the baseline must be rejected, not applied at the line number written in its header.
Run: python test_apply_delta.py   (also collected by pytest)"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("DeltaUtil", Path(__file__).parent / "ab/gpt/util/nn/DeltaUtil.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

BASE = """import torch


class Net:
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x

    def learn(self, data):
        outputs = self(data)
        loss = outputs.sum()
        return loss
"""

# The LLM copied a context/removed line that is not in the baseline ("self.c"), and its header points
# into learn(). Neither the header normalisation nor `patch` can place this hunk.
BAD = """--- baseline.py
+++ improved.py
@@ -10,3 +10,3 @@
     def forward(self, x):
-        x = self.c(x)
+        x = self.d(x)
         x = self.b(x)
"""

GOOD = """--- baseline.py
+++ improved.py
@@ -5,3 +5,3 @@
     def forward(self, x):
-        x = self.a(x)
+        x = self.d(x)
         x = self.b(x)
"""


def test_unplaceable_hunk_is_rejected():
    out = D.apply_delta(BASE, BAD)
    assert out is None, "hunk applied at the wrong place:\n" + str(out)


def test_correct_hunk_still_applies():
    expected = BASE.replace("x = self.a(x)", "x = self.d(x)")
    assert D.apply_delta(BASE, GOOD).rstrip("\n") == expected.rstrip("\n")
    assert D._apply_delta_manual(BASE, GOOD).rstrip("\n") == expected.rstrip("\n")


def test_multi_hunk_roundtrip_manual():
    improved = BASE.replace("x = self.a(x)", "x = self.a(x)\n        x = self.relu(x)").replace("loss = outputs.sum()", "loss = outputs.mean()")
    delta = D.compute_delta(BASE, improved)
    assert D._apply_delta_manual(BASE, delta).rstrip("\n") == improved.rstrip("\n")
    assert D.apply_delta(BASE, delta).rstrip("\n") == improved.rstrip("\n")


if __name__ == "__main__":
    for t in (test_unplaceable_hunk_is_rejected, test_correct_hunk_still_applies, test_multi_hunk_roundtrip_manual):
        try:
            t()
            print("PASS", t.__name__)
        except AssertionError as e:
            print("FAIL", t.__name__, "-", e)
