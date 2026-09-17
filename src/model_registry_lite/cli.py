"""Command-line interface for model-registry-lite.

Usage examples:

    model-registry register --name churn-model --version 1.0.0 \\
        --framework sklearn --metrics '{"f1": 0.91, "auc": 0.96}' \\
        --hyperparameters '{"max_depth": 8}' --tags tabular,baseline

    model-registry promote --name churn-model --version 1.0.0 \\
        --by "Anusha Mukka" --note "Beat baseline on holdout set"

    model-registry list --stage production
    model-registry search --metric "f1>=0.9" --framework sklearn
    model-registry export --out registry-backup.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import __version__
from .registry import ModelRegistry, RegistryError

DEFAULT_DB = "model_registry.db"


def _kv_json(text: str, label: str) -> dict:
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: --{label} is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise SystemExit(f"error: --{label} must be a JSON object.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-registry",
        description="A minimal model registry with versioning, stage approvals, and metric search.",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to the SQLite registry file.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Register a new model version.")
    reg.add_argument("--name", required=True)
    reg.add_argument("--version", required=True, dest="model_version")
    reg.add_argument("--framework", default="")
    reg.add_argument("--description", default="")
    reg.add_argument("--artifact-path", default="")
    reg.add_argument("--dataset-hash", default="")
    reg.add_argument("--metrics", default="{}", help='JSON object, e.g. \'{"f1": 0.91}\'')
    reg.add_argument("--hyperparameters", default="{}", help="JSON object of hyperparameters.")
    reg.add_argument("--tags", default="", help="Comma-separated tags.")
    reg.add_argument("--overwrite", action="store_true", help="Re-register an existing version.")

    upd = sub.add_parser("update", help="Update metadata of a registered version.")
    upd.add_argument("--name", required=True)
    upd.add_argument("--version", required=True, dest="model_version")
    upd.add_argument("--metrics", help='JSON object, e.g. \'{"f1": 0.92}\'')
    upd.add_argument("--hyperparameters", help="JSON object.")
    upd.add_argument("--tags", help="Comma-separated tags (replaces existing).")
    upd.add_argument("--description", help="New description.")

    show = sub.add_parser("show", help="Show full metadata for one model version.")
    show.add_argument("--name", required=True)
    show.add_argument("--version", required=True, dest="model_version")

    lst = sub.add_parser("list", help="List registered models, optionally by stage.")
    lst.add_argument("--stage", choices=["staging", "production", "archived"], default=None)
    lst.add_argument("--json", action="store_true", help="Print full JSON documents.")

    ver = sub.add_parser("versions", help="List all versions of a model.")
    ver.add_argument("--name", required=True)

    for name, help_text in (
        ("promote", "Move a version from staging to production."),
        ("archive", "Move a version to archived."),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--name", required=True)
        cmd.add_argument("--version", required=True, dest="model_version")
        cmd.add_argument("--by", required=True, help="Approver name (recorded in the audit trail).")
        cmd.add_argument("--note", default="", help="Approval note.")

    hist = sub.add_parser("history", help="Show the stage-transition audit trail.")
    hist.add_argument("--name", required=True)
    hist.add_argument("--version", required=True, dest="model_version")

    search = sub.add_parser("search", help="Search models by metadata and metric thresholds.")
    search.add_argument("--name-contains", default=None)
    search.add_argument("--framework", default=None)
    search.add_argument("--stage", choices=["staging", "production", "archived"], default=None)
    search.add_argument("--tag", action="append", default=[], help="Repeatable; all tags must match.")
    search.add_argument(
        "--metric",
        action="append",
        default=[],
        help='Repeatable; e.g. --metric "f1>=0.9". All filters must match.',
    )
    search.add_argument("--json", action="store_true")

    exp = sub.add_parser("export", help="Export the whole registry as JSON.")
    exp.add_argument("--out", default=None, help="Write to file; prints to stdout if omitted.")

    imp = sub.add_parser("import", help="Import a registry JSON export.")
    imp.add_argument("--in", required=True, dest="input", help="Path to the JSON export file.")
    imp.add_argument("--overwrite", action="store_true")

    rm = sub.add_parser("deregister", help="Delete a model version from the registry.")
    rm.add_argument("--name", required=True)
    rm.add_argument("--version", required=True, dest="model_version")
    rm.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    return parser


def _print_summary(model) -> None:
    metrics = ", ".join(f"{k}={v}" for k, v in model.metrics.items()) or "-"
    print(f"{model.name}:{model.version}  [{model.stage}]  {model.framework}  metrics({metrics})")


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = ModelRegistry(args.db)
    try:
        return _dispatch(registry, args)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        registry.close()


def _dispatch(registry: ModelRegistry, args: argparse.Namespace) -> int:
    if args.command == "register":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        model = registry.register(
            name=args.name,
            version=args.model_version,
            framework=args.framework,
            description=args.description,
            artifact_path=args.artifact_path,
            dataset_hash=args.dataset_hash,
            metrics=_kv_json(args.metrics, "metrics"),
            hyperparameters=_kv_json(args.hyperparameters, "hyperparameters"),
            tags=tags,
            overwrite=args.overwrite,
        )
        _print_summary(model)
        return 0

    if args.command == "update":
        fields = {}
        if args.metrics is not None:
            fields["metrics"] = _kv_json(args.metrics, "metrics")
        if args.hyperparameters is not None:
            fields["hyperparameters"] = _kv_json(args.hyperparameters, "hyperparameters")
        if args.tags is not None:
            fields["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.description is not None:
            fields["description"] = args.description
        if not fields:
            print("error: nothing to update; pass at least one of --metrics/--hyperparameters/--tags/--description",
                  file=sys.stderr)
            return 2
        _print_summary(registry.update_metadata(args.name, args.model_version, **fields))
        return 0

    if args.command == "show":
        print(registry.get(args.name, args.model_version).to_json())
        return 0

    if args.command == "list":
        models = registry.list_models(stage=args.stage)
        if args.json:
            print(json.dumps([m.to_dict() for m in models], indent=2, sort_keys=True))
        else:
            for model in models:
                _print_summary(model)
            print(f"({len(models)} model version(s))")
        return 0

    if args.command == "versions":
        for model in registry.list_versions(args.name):
            _print_summary(model)
        return 0

    if args.command in ("promote", "archive"):
        action = registry.promote if args.command == "promote" else registry.archive
        model = action(args.name, args.model_version, approved_by=args.by, note=args.note)
        _print_summary(model)
        return 0

    if args.command == "history":
        entries = registry.store.transition_history(args.name, args.model_version)
        if not entries:
            print("No recorded transitions.")
        for entry in entries:
            note = f" — {entry.note}" if entry.note else ""
            print(f"{entry.at}: {entry.from_stage} -> {entry.to_stage} by {entry.approved_by}{note}")
        return 0

    if args.command == "search":
        results = registry.search(
            name_contains=args.name_contains,
            framework=args.framework,
            stage=args.stage,
            tags=args.tag or None,
            metric_filters=args.metric or None,
        )
        if args.json:
            print(json.dumps([m.to_dict() for m in results], indent=2, sort_keys=True))
        else:
            for model in results:
                _print_summary(model)
            print(f"({len(results)} match(es))")
        return 0

    if args.command == "export":
        payload = registry.export_json()
        if args.out:
            registry.export_to_file(args.out)
            print(f"Exported registry to {args.out}")
        else:
            print(payload)
        return 0

    if args.command == "import":
        imported, skipped = registry.import_from_file(args.input, overwrite=args.overwrite)
        print(f"Imported {imported} model version(s), skipped {skipped} existing.")
        return 0

    if args.command == "deregister":
        if not args.yes:
            answer = input(f"Delete {args.name}:{args.model_version}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 2
        registry.deregister(args.name, args.model_version)
        print(f"Deleted {args.name}:{args.model_version}.")
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
