from __future__ import annotations

import argparse
import json

from fastwam_ood_eval.cli import _doctor


def test_doctor_reports_checkout_backends(capsys):
    status = _doctor(argparse.Namespace(config=None, set=[]))

    assert status == 0
    report = json.loads(capsys.readouterr().out)
    assert report["packages"]["libero"].startswith("available via checkout adapter")
