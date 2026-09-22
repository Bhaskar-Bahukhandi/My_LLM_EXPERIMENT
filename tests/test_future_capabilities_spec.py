"""Document integrity only: no model import, data access or runtime activation."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/future_capabilities_contract_v1.json"
FINAL_HASHES = {
    "Unified_Edge400_Master_Bible_v3.0_FINAL.md":
        "754e99e9feca52c8744b71891d6e71993233dea43c3d7155900334e28c121f75",
    "Unified_Edge400_Master_Roadmap_and_Formula_Board_v1.0_FINAL.md":
        "91bb81b09162e8ccdbdd2baa5bcd5c849a912bfe15e53f798c09bd70b788af99",
}


def test_contract_enumerations_and_no_activation():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema_version"] == contract["addendum_version"] == "1.0"
    assert contract["artifact_role"] == "SPECIFICATION_ONLY_NOT_RUNTIME_CONFIG"
    assert contract["effort_levels"] == ["Auto", "Low", "Medium", "High", "Ultra"]
    assert contract["rsi_maturity_levels"] == [f"RSI-{i}" for i in range(5)]
    assert contract["promotion_outcomes"] == ["PROMOTE", "HOLD", "ROLLBACK", "REJECT"]
    assert contract["context_statuses"] == ["VERIFIED", "PARTIAL", "UNVERIFIED"]
    for value in contract.values():
        if isinstance(value, list):
            assert len(value) == len(set(value)), value
    scopes = contract["resource_bounded_requirement"]["scopes"]
    assert len(scopes) == len(set(scopes)) == 3
    assert set(contract["feature_states"].values()) == {"PROPOSED_NOT_IMPLEMENTED_NOT_ENABLED"}
    assert contract["private_reasoning_disclosure_required"] is False
    assert contract["base_path_non_regression"]["off_preserves_base"] is True
    assert contract["base_path_non_regression"]["activation_requires_separate_authorization"]
    assert contract["resource_bounded_requirement"]["mandatory"] is True
    assert contract["resource_bounded_requirement"]["nested_budget_reset_permitted"] is False
    assert {
        "overwrite_parent", "modify_evaluator", "access_or_modify_hidden_evaluation_data",
        "modify_promotion_thresholds", "increase_resource_allowance", "self_promote",
        "write_production_code", "modify_host_permissions",
    } <= set(contract["prohibited_candidate_authorities"])


def test_lineage_evaluator_and_context_fields():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert set(contract["generation_manifest_required_fields"]) == set(
        "schema_version generation_id parent_generation_id parent_checkpoint_sha256 "
        "proposal_id source_commit model_config_sha256 training_config_sha256 "
        "dataset_manifest_sha256 evaluator_manifest_sha256 hidden_evaluator_version "
        "compute_budget experiment_budget capability_metrics regression_metrics "
        "efficiency_metrics security_metrics promotion_decision promotion_authority "
        "rollback_parent creation_timestamp".split()
    )
    assert set(contract["context_certification_fields"]) == set(
        "mechanically_supported training_window_bytes validation_window_bytes "
        "maximum_evaluated_context_bytes effective_context_evidence retention_tests status".split()
    )
    assert set(contract["evaluator_categories"]) == set(
        "general_capability language_modeling_quality code mathematics long_context_retention "
        "retrieval_rag tool_use verifier_repair_success latency throughput memory "
        "energy_resource_cost_where_measurable security robustness regression contamination "
        "benchmark_integrity".split()
    )


def test_final_hashes_and_proposal_links():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["final_specification_sha256"] == FINAL_HASHES
    for name, checksum in FINAL_HASHES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum
    addendum = ROOT / "Unified_Edge400_Future_Capabilities_Addendum_v1.0.md"
    text = addendum.read_text(encoding="utf-8")
    assert len(contract["proposal_documents"]) == 3
    for number, name in zip((6, 7, 8), contract["proposal_documents"], strict=True):
        assert f"AC-{number:03d}" in name and name in text
    for path in [addendum] + [ROOT / name for name in contract["proposal_documents"]]:
        content = path.read_text(encoding="utf-8")
        assert "PROPOSED" in content
        assert content.count("```") % 2 == 0
        for target in re.findall(r"\]\(([^)]+)\)", content):
            assert (path.parent / target).is_file(), (path.name, target)
