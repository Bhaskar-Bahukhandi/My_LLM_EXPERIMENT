"""Review-only support-field safeguard; never used by corpus admission."""

import ast

from corpus_benchmark_index import METADATA_FIELDS
from corpus_contamination import ExclusionIndex


def isolated_import(text):
    """Accept only one ordinary Python import statement, without executing it."""
    try:
        tree = ast.parse(text.strip())
    except (SyntaxError, ValueError):
        return False
    return len(tree.body) == 1 and isinstance(tree.body[0], (ast.Import, ast.ImportFrom))


def review_fields(fields):
    """Keep every nonmetadata field except syntactic imports in test_imports.

    The caller supplies the original structured field path, never a guessed path
    extracted from an identity string. Answers remain protected even if their
    bytes also occur in a suppressed support field.
    """
    for field in fields:
        root = field["field_path"].strip("/").split("/")[0]
        if root in METADATA_FIELDS:
            continue
        if root == "test_imports" and isolated_import(field["text"]):
            continue
        yield field


class ProposedIndex(ExclusionIndex):
    """Unadopted v2 prototype, preserving v1 exact and near matching otherwise."""

    def __init__(self, fields):
        super().__init__(review_fields(fields))
