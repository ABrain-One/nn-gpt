"""Tests for ab.gpt.brute.loss_opt.VariantGen.

Covers the loss/optimizer substitution: the supported catalogue, the NGL
optimizer restriction (Adam/AdamW only), NGL class injection, and that the
removed NLLLoss is now rejected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ab.gpt.brute.loss_opt.VariantGen import make_variant, iter_variants, LOSS_SPECS, OPTIM_SPECS

SIMPLE_SRC = '''\
import torch
from torch import nn

class SimpleNet(nn.Module):
    def train_setup(self, prm):
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'])

    def learn(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criteria(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @staticmethod
    def supported_hyperparameters():
        return {'lr'}
'''

TUPLE_SRC = '''\
import torch
from torch import nn

class TupleNet(nn.Module):
    def train_setup(self, prm):
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm.get('momentum', 0.9))

    def learn(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criteria[0](outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @staticmethod
    def supported_hyperparameters():
        return {'lr', 'momentum'}
'''


# ---------- catalogue ----------
def test_nllloss_removed():
    assert "NLLLoss" not in LOSS_SPECS
    src, err = make_variant(SIMPLE_SRC, "NLLLoss", "Adam")
    assert src is None and err is not None, "NLLLoss should now be rejected"


def test_supported_losses():
    assert set(LOSS_SPECS) == {"CrossEntropyLoss", "NGL"}


def test_no_log_softmax_anywhere():
    # log_softmax machinery existed only for NLLLoss; nothing should inject it now.
    for loss in LOSS_SPECS:
        for opt in ("Adam",):
            src, err = make_variant(SIMPLE_SRC, loss, opt)
            assert err is None, err
            assert "F.log_softmax" not in src


# ---------- CrossEntropyLoss ----------
def test_cel_substitution():
    src, err = make_variant(SIMPLE_SRC, "CrossEntropyLoss", "RMSprop")
    assert err is None, err
    assert "nn.CrossEntropyLoss(" in src
    assert "torch.optim.RMSprop(" in src


def test_cel_tuple_form_preserved():
    src, err = make_variant(TUPLE_SRC, "CrossEntropyLoss", "Adam")
    assert err is None, err
    assert "self.criteria = (nn.CrossEntropyLoss" in src  # stays a tuple


# ---------- NGL restriction ----------
def test_ngl_allowed_with_adam():
    for opt in ("Adam", "AdamW"):
        src, err = make_variant(SIMPLE_SRC, "NGL", opt)
        assert err is None, f"NGL+{opt} should be allowed: {err}"
        assert "class NGL(nn.Module)" in src, "NGL class should be injected"


def test_ngl_rejected_with_other_optimizers():
    for opt in ("SGD", "RMSprop", "Adagrad", "Adadelta"):
        src, err = make_variant(SIMPLE_SRC, "NGL", opt)
        assert src is None and err is not None, f"NGL+{opt} should be rejected"
        assert "restricted" in err


# ---------- grid ----------
def test_unknown_loss_returns_error():
    src, err = make_variant(SIMPLE_SRC, "BogusLoss", "Adam")
    assert src is None and err is not None


def test_iter_variants_skips_disallowed_ngl_combos():
    # Full grid: CEL works with every optimizer; NGL only with Adam/AdamW.
    results = list(iter_variants(SIMPLE_SRC))
    ok = [(l, o) for l, o, s, e in results if e is None]
    bad = [(l, o) for l, o, s, e in results if e is not None]

    n_opt = len(OPTIM_SPECS)
    assert len(ok) == n_opt + 2          # CEL×all + NGL×{Adam, AdamW}
    assert all(l == "NGL" for l, o in bad)
    assert {o for l, o in ok if l == "NGL"} == {"Adam", "AdamW"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)
