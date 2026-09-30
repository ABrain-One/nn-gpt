"""Tests for ab.gpt.brute.loss_opt.VariantGen.

Covers the loss/optimizer substitution: the supported catalogue, NGL class
injection across the full optimizer grid, and that the removed NLLLoss is
now rejected.
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


# ---------- NGL ----------
def test_ngl_works_with_every_optimizer():
    for opt in OPTIM_SPECS:
        src, err = make_variant(SIMPLE_SRC, "NGL", opt)
        assert err is None, f"NGL+{opt} should be allowed: {err}"
        assert "class NGL(nn.Module)" in src, "NGL class should be injected"


# ---------- grid ----------
def test_unknown_loss_returns_error():
    src, err = make_variant(SIMPLE_SRC, "BogusLoss", "Adam")
    assert src is None and err is not None


def test_iter_variants_covers_full_grid():
    # Every loss pairs with every optimizer — no skipped combinations.
    results = list(iter_variants(SIMPLE_SRC))
    ok = [(l, o) for l, o, s, e in results if e is None]
    bad = [(l, o) for l, o, s, e in results if e is not None]

    assert len(ok) == len(LOSS_SPECS) * len(OPTIM_SPECS)
    assert not bad


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
