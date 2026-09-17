"""End-to-end tests for the command-line interface."""

import json

import pytest

from model_registry_lite.cli import main


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "cli.db")


def test_cli_register_list_search_promote(capsys, db):
    assert main(["--db", db, "register", "--name", "churn", "--version", "1.0.0",
                 "--framework", "sklearn", "--metrics", '{"f1": 0.91}',
                 "--tags", "tabular,baseline"]) == 0

    assert main(["--db", db, "list"]) == 0
    out = capsys.readouterr().out
    assert "churn:1.0.0" in out and "[staging]" in out

    assert main(["--db", db, "promote", "--name", "churn", "--version", "1.0.0",
                 "--by", "Anusha Mukka", "--note", "QA pass"]) == 0
    out = capsys.readouterr().out
    assert "[production]" in out

    assert main(["--db", db, "search", "--metric", "f1>=0.9", "--framework", "sklearn"]) == 0
    out = capsys.readouterr().out
    assert "churn:1.0.0" in out

    assert main(["--db", db, "search", "--metric", "f1>=0.99"]) == 0
    assert "(0 match(es))" in capsys.readouterr().out


def test_cli_show_and_history(capsys, db):
    main(["--db", db, "register", "--name", "churn", "--version", "1.0.0"])
    main(["--db", db, "promote", "--name", "churn", "--version", "1.0.0",
          "--by", "Anusha Mukka", "--note", "QA pass"])
    capsys.readouterr()  # flush earlier command output

    assert main(["--db", db, "show", "--name", "churn", "--version", "1.0.0"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["stage"] == "production"

    assert main(["--db", db, "history", "--name", "churn", "--version", "1.0.0"]) == 0
    out = capsys.readouterr().out
    assert "staging -> production" in out and "Anusha Mukka" in out


def test_cli_export_import_round_trip(capsys, db, tmp_path):
    main(["--db", db, "register", "--name", "churn", "--version", "1.0.0",
          "--metrics", '{"f1": 0.91}'])
    export_file = str(tmp_path / "export.json")
    assert main(["--db", db, "export", "--out", export_file]) == 0

    fresh = str(tmp_path / "fresh.db")
    assert main(["--db", fresh, "import", "--in", export_file]) == 0
    out = capsys.readouterr().out
    assert "Imported 1" in out
    assert main(["--db", fresh, "list", "--json"]) == 0
    docs = json.loads(capsys.readouterr().out)
    assert docs[0]["metrics"]["f1"] == 0.91


def test_cli_errors_are_user_friendly(capsys, db):
    main(["--db", db, "register", "--name", "churn", "--version", "1.0.0"])
    # duplicate without --overwrite
    rc = main(["--db", db, "register", "--name", "churn", "--version", "1.0.0"])
    assert rc == 1
    assert "already exists" in capsys.readouterr().err
    # invalid JSON metrics -> SystemExit with a user-facing message
    with pytest.raises(SystemExit) as exc_info:
        main(["--db", db, "register", "--name", "x", "--version", "1.0.0",
              "--metrics", "{nope"])
    assert "not valid JSON" in str(exc_info.value.code)
    # promote requires an approver
    with pytest.raises(SystemExit):
        main(["--db", db, "promote", "--name", "churn", "--version", "1.0.0"])


def test_cli_deregister_with_flag(capsys, db):
    main(["--db", db, "register", "--name", "churn", "--version", "1.0.0"])
    assert main(["--db", db, "deregister", "--name", "churn", "--version", "1.0.0",
                 "--yes"]) == 0
    assert "Deleted" in capsys.readouterr().out
    assert main(["--db", db, "list"]) == 0
    assert "(0 model version(s))" in capsys.readouterr().out
