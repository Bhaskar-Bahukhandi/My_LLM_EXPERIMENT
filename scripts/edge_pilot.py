"""Narrow local R9 CLI. Read-only by default; no host/Kaggle integration."""

import argparse
import json
import signal
from contextlib import contextmanager
from pathlib import Path

from unified_edge.resolve import ResolvedConfig
from unified_edge.training.config import TrainingConfig, canonical_hash
from unified_edge.training.data import DatasetManifest
from unified_edge.training.pilot_context import AnchorManifest
from unified_edge.training.pilot_data import ValidationSubsets, validate_capacity
from unified_edge.training.pilot_gates import GateReceipt, validate_prerequisites
from unified_edge.training.pilot_plan import MODEL_SHA, PilotPlan, require
from unified_edge.training.pilot_runner import PilotRunner, read_record, subject_identity


@contextmanager
def cooperative_stops(runner):
    """SIGINT/SIGTERM request a stop; the current update can reach its safe boundary."""
    previous = {}
    try:
        for event in (signal.SIGINT, signal.SIGTERM):
            previous[event] = signal.signal(event, lambda *_: runner.request_stop())
        yield
    finally:
        for event, handler in previous.items():
            signal.signal(event, handler)


def load_bundle(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(
        set(value)
        == {
            "plan",
            "model",
            "config",
            "data",
            "subsets",
            "anchors",
            "receipts",
            "selected_lr_receipt_sha256",
        },
        "invalid execution bundle",
    )
    return {
        "plan": PilotPlan.from_dict(value["plan"]),
        "model": ResolvedConfig.from_dict(value["model"]),
        "config": TrainingConfig.from_dict(value["config"]),
        "data": DatasetManifest.from_dict(value["data"]),
        "subsets": ValidationSubsets.from_dict(value["subsets"]),
        "anchors": AnchorManifest.from_dict(value["anchors"]),
        "receipts": tuple(GateReceipt.from_dict(r) for r in value["receipts"]),
        "selected_lr_receipt_sha256": value["selected_lr_receipt_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("plan", "inspect", "validate-prerequisites", "launch", "resume")
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--run-record", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--trusted-receipt-sha", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(
        not args.execute or args.action in ("launch", "resume"),
        "execution requires explicit action",
    )
    if args.action == "plan":
        plan = PilotPlan()
        result = {
            "plan": plan.to_dict(),
            "plan_sha256": plan.sha256,
            "lr_probe_sha256": plan.lr_probe.sha256,
            "nominal_target_bytes": plan.updates * plan.nominal_bytes_per_update,
            "cadence": {
                str(i): plan.evaluation.events(i) for i in range(8001) if plan.evaluation.events(i)
            },
            "execution_authorized": False,
            "execution_started": False,
        }
    elif args.action == "inspect":
        require(args.run_record is not None, "--run-record required")
        result = read_record(args.run_record)
    else:
        require(args.bundle is not None, "--bundle required")
        bundle = load_bundle(args.bundle)
        plan, model, data = bundle["plan"], bundle["model"], bundle["data"]
        require(
            model.sha256 == MODEL_SHA
            and bundle["config"] == plan.training_config(bundle["config"].learning_rate, "cuda:0"),
            "real bundle model/config mismatch",
        )
        bundle["subsets"].validate(data)
        bundle["anchors"].validate(data)
        validate_capacity(data)
        subject = subject_identity(plan, model, data, bundle["subsets"], bundle["anchors"])
        result = validate_prerequisites(
            bundle["receipts"],
            canonical_hash(subject),
            trusted_receipt_shas=tuple(args.trusted_receipt_sha),
        )
        if args.execute:
            require(
                result["status"] == "PASS" and args.root is not None and args.data_root is not None,
                "approved prerequisites and explicit paths required",
            )
            runner = PilotRunner(
                args.root,
                **bundle,
                data_root=args.data_root,
                trusted_receipt_shas=tuple(args.trusted_receipt_sha),
                resume=args.action == "resume",
            )
            try:
                with cooperative_stops(runner):
                    result = runner.run()
            finally:
                runner.close()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("status") != "INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
