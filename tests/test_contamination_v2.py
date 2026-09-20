"""All-identity scanning preserves answer protection after ignored support fields."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from corpus_contamination_v2 import ExhaustiveIndex  # noqa: E402


def field(text, path):
    return {"identity": "benchmark/test/1" + path, "field_path": path, "text": text}


def test_exhaustive_scan_records_ignored_import_and_every_protected_field():
    fields = [field("import math", "/test_imports/0")]
    for path in ("/problem", "/answer", "/canonical_solution", "/175b_verification/solution"):
        fields.append(
            field("Distinctive content associated with " + path + " and lattice geometry.", path)
        )
    payload = "\n\n".join(f["text"] for f in fields)
    index = ExhaustiveIndex(fields)
    result = index.match_all(payload)
    assert len(result) == 5
    assert sum(r["actionable"] for r in result) == 4
    assert result == ExhaustiveIndex(reversed(fields)).match_all(payload)
    alone = index.match_all("import math")
    assert len(alone) == 1 and not alone[0]["actionable"]
    assert sum(r["actionable"] for r in index.match_all(payload.upper())) == 4


def test_short_complete_answers_and_near_matches_survive_support_exception():
    answer = " ".join(f"word{i}" for i in range(100))
    index = ExhaustiveIndex(
        [
            field("import math", "/test_imports/0"),
            field("return 1", "/canonical_solution"),
            field(answer, "/answer"),
        ]
    )
    result = index.match_all("import math\nreturn 1\n\n" + answer.replace("word50", "changed"))
    assert len(result) == 3 and sum(r["actionable"] for r in result) == 2
    assert any("JACCARD" in r["method"] for r in result)
    both = ExhaustiveIndex(
        [field("import math", "/test_imports/0"), field("import math", "/answer")]
    ).match_all("import math")
    assert both[0]["actionable"]


def test_metadata_is_omitted_and_non_import_support_remains_protected():
    for root in ("task_id", "entry_point", "source_file", "level", "type"):
        assert (
            ExhaustiveIndex([field("private metadata", "/" + root)]).match_all("private metadata")
            == []
        )
    support = "import math\nassert True"
    assert ExhaustiveIndex([field(support, "/test_imports/0")]).match_all(support)[0]["actionable"]
