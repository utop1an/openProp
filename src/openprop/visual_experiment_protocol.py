from __future__ import annotations

import hashlib
import json
from pathlib import Path


VISUAL_EXPERIMENT_PROTOCOL = "openprop-iclr2027-visual-experiment-v2"
AI2THOR_SPLIT_SHA256 = "c99adcdc6f7118f11c2bfdd66143beb71279665668c406eb18c92bdfe2d80ea4"


def build_visual_experiment_protocol() -> dict[str, object]:
    """Return the preregistered matrix without using model outputs or test truth."""

    payload: dict[str, object] = {
        "schema_version": 2,
        "protocol_id": VISUAL_EXPERIMENT_PROTOCOL,
        "target_venue": "ICLR 2027",
        "selection_uses_model_outputs": False,
        "selection_uses_test_truth": False,
        "evidence_status": "protocol_only_until_captured_responses_exist",
        "ai2thor_scene_split": {
            "path": "artifacts/ai2thor_protocol/scene_split.json",
            "protocol_id": "openprop-ai2thor-scene-split-v1",
            "split_sha256": AI2THOR_SPLIT_SHA256,
            "scene_counts": {"development": 72, "calibration": 24, "test": 24},
            "cluster_unit": "scene",
        },
        "random_seeds": [1901, 2903, 3907, 4909, 5923],
        "language_parser_freeze_rule": {
            "main_parser_count": 1,
            "robustness_parser_count": 1,
            "reuse_identical_parse_across_visual_systems": True,
            "exact_provider_model_revision_prompt_schema_and_settings_required": True,
            "lock_before_calibration_or_test_inference": True,
            "baselines": ["deterministic_rule_parser", "oracle_typed_constraint"],
            "required_metrics": [
                "schema_valid_rate", "typed_exact_match", "typed_macro_f1",
                "relevance_brier", "relevance_ece", "paraphrase_consistency"
            ],
            "missing_or_malformed_output": "retained_as_failure",
            "current_status": "exact_models_not_yet_locked",
        },
        "vlm_freeze_rule": {
            "minimum_model_families": 2,
            "exact_provider_model_revision_and_request_settings_required": True,
            "lock_before_calibration_or_test_inference": True,
            "reuse_identical_captured_inputs": True,
            "missing_or_malformed_output": "retained_as_failure_or_abstention",
            "current_status": "exact_models_not_yet_locked",
        },
        "model_factorization": {
            "main_matrix": "one_frozen_llm_parser_x_two_vlm_families_x_all_systems",
            "parser_robustness": "second_parser_x_one_vlm_x_main_and_oracle_parser",
            "vlm_robustness": "oracle_typed_query_x_two_vlm_families",
            "ablation_replay": "all_systems_reuse_identical_captured_model_outputs",
            "main_system_vlm_does_not_rank_final_entities": True,
        },
        "evidence_tiers": [
            {
                "id": "ai2thor_ithor",
                "role": "controlled_causal_end_to_end",
                "end_to_end_query_claim_eligible": True,
                "cluster_unit": "scene",
                "status": "capture_blocked_pending_supported_gpu_linux",
            },
            {
                "id": "procthor_ood",
                "role": "simulator_layout_appearance_transfer",
                "end_to_end_query_claim_eligible": True,
                "cluster_unit": "house",
                "status": "adapter_and_capture_pending",
            },
            {
                "id": "custom_real_video",
                "role": "real_world_end_to_end_confirmation",
                "end_to_end_query_claim_eligible": True,
                "cluster_unit": "room_person",
                "status": "collection_pending",
            },
            {
                "id": "ego4d_hands_objects",
                "role": "state_change_detection_and_changed_object_association",
                "end_to_end_query_claim_eligible": False,
                "cluster_unit": "video_participant",
                "status": "access_and_adapter_pending",
            },
            {
                "id": "aria_digital_twin",
                "role": "candidate_tracking_occlusion_and_moved_object_identity",
                "end_to_end_query_claim_eligible": False,
                "cluster_unit": "sequence",
                "status": "access_and_adapter_pending",
            },
            {
                "id": "epic_kitchens_visor",
                "role": "candidate_generation_and_appearance_shift",
                "end_to_end_query_claim_eligible": False,
                "cluster_unit": "participant_video",
                "status": "access_and_adapter_pending",
            },
            {
                "id": "licensed_web_video",
                "role": "adversarial_camera_and_source_shift_only",
                "end_to_end_query_claim_eligible": False,
                "cluster_unit": "creator_video",
                "status": "rights_screening_and_annotation_pending",
            },
        ],
        "ai2thor_factor_grid": {
            "changed_objects": [0, 1, 2, 3],
            "same_type_distractors": [0, 1, 3, 7],
            "target_visibility": ["visible", "partially_occluded", "absent"],
            "camera": ["fixed", "translated", "rotated"],
            "history_gap_seconds": [0, 300, 3600, 86400],
            "change_family": [
                "move_receptacle", "open", "toggle", "dirty", "fill", "cook", "slice", "break"
            ],
            "query_wording": ["canonical", "paraphrase_a", "paraphrase_b"],
            "candidate_source": ["oracle_boxes", "detected_tracked_boxes"],
            "design": "deterministic_balanced_fractional_factorial_not_full_cartesian",
        },
        "sampling": {
            "ai2thor_minimum_successful_episodes_per_scene_per_seed": 4,
            "ai2thor_failed_actions_retained_in_construction_denominator": True,
            "custom_real_video_minimum_test_room_person_clusters": 8,
            "custom_real_video_minimum_test_episodes_per_cluster": 6,
            "final_sample_size_rule": "increase_only_from_blinded_pilot paired-effect variance before test inference",
            "development_use": "engineering_only",
            "calibration_use": "all learned policies and thresholds only",
            "test_use": "single untouched confirmation",
        },
        "systems": [
            {"id": "current_frame_vlm", "kind": "baseline", "main_claim_eligible": True},
            {"id": "direct_vlm_updater", "kind": "baseline", "main_claim_eligible": True},
            {"id": "latest_accepted_observation", "kind": "baseline", "main_claim_eligible": True},
            {"id": "openprop_no_decay", "kind": "baseline", "main_claim_eligible": True},
            {"id": "openprop_fixed_decay", "kind": "baseline", "main_claim_eligible": True},
            {"id": "openprop_learned_global", "kind": "main", "main_claim_eligible": True},
            {"id": "openprop_no_query_score", "kind": "ablation", "main_claim_eligible": False},
            {"id": "openprop_no_region_anchor", "kind": "ablation", "main_claim_eligible": False},
            {"id": "openprop_no_track_evidence", "kind": "ablation", "main_claim_eligible": False},
            {"id": "openprop_no_null", "kind": "ablation", "main_claim_eligible": False},
            {"id": "openprop_no_margin", "kind": "ablation", "main_claim_eligible": False},
            {"id": "openprop_pooled_source_reliability", "kind": "ablation", "main_claim_eligible": False},
            {"id": "openprop_independent_assignment", "kind": "ablation", "main_claim_eligible": False},
            {"id": "openprop_rule_parser", "kind": "language_baseline", "main_claim_eligible": False},
            {"id": "openprop_oracle_parser", "kind": "upper_bound", "main_claim_eligible": False},
            {"id": "oracle_candidates", "kind": "upper_bound", "main_claim_eligible": False},
            {"id": "oracle_properties", "kind": "upper_bound", "main_claim_eligible": False},
            {"id": "oracle_identity", "kind": "upper_bound", "main_claim_eligible": False},
        ],
        "primary_comparisons": [
            {
                "id": "main_vs_current_frame",
                "system": "openprop_learned_global",
                "baseline": "current_frame_vlm",
                "endpoint": "query_top1_all_cases",
                "direction": "higher",
                "multiplicity_family": "primary_query",
            },
            {
                "id": "main_vs_direct_updater",
                "system": "openprop_learned_global",
                "baseline": "direct_vlm_updater",
                "endpoint": "query_top1_all_cases",
                "direction": "higher",
                "multiplicity_family": "primary_query",
            },
            {
                "id": "main_vs_latest_observation",
                "system": "openprop_learned_global",
                "baseline": "latest_accepted_observation",
                "endpoint": "query_top1_all_cases",
                "direction": "higher",
                "multiplicity_family": "primary_query",
            },
            {
                "id": "main_vs_no_decay",
                "system": "openprop_learned_global",
                "baseline": "openprop_no_decay",
                "endpoint": "query_top1_all_cases",
                "direction": "higher",
                "multiplicity_family": "primary_query",
            },
            {
                "id": "global_vs_independent_crowded",
                "system": "openprop_learned_global",
                "baseline": "openprop_independent_assignment",
                "endpoint": "false_update_rate_all_detections",
                "direction": "lower",
                "slice": "changed_objects>=2 or same_type_distractors>=3",
                "multiplicity_family": "primary_safety",
            },
        ],
        "secondary_comparisons": [
            "openprop_learned_global_vs_openprop_fixed_decay",
            "all_component_ablations_vs_openprop_learned_global",
            "detected_tracked_boxes_vs_oracle_candidates",
            "per_vlm_family_effects",
            "per_property_source_visibility_delay_and_candidate_count_slices",
        ],
        "calibration_gates": {
            "candidate_minimum_recall": 0.90,
            "candidate_maximum_identity_switch_rate": 0.05,
            "association_maximum_false_update_rate": 0.0,
            "query_maximum_false_answer_rate": 0.0,
            "unseen_candidate_count": "abstain",
            "source_specific_confidence_map": "requires_calibration_support_else_global_fallback",
        },
        "analysis": {
            "primary_unit": "delayed_language_query",
            "all_failures_and_abstentions_retained": True,
            "point_estimate": "pooled_explicit_numerators_and_denominators",
            "uncertainty": "paired_cluster_bootstrap_95_percent",
            "primary_multiplicity": "simultaneous_intervals_within_declared_family",
            "binary_paired_test": "exact_mcnemar",
            "continuous_or_rank_paired_test": "exact_sign",
            "minimum_simulator_seeds": 5,
            "required_outputs": [
                "main_table", "decomposition_table", "horizon_table", "ablation_table",
                "transfer_table", "failure_safety_table", "reliability_plots",
                "risk_coverage_plot", "delay_plot", "distractor_plot", "candidate_plot",
                "sample_efficiency_plot", "source_property_forest", "confusion_matrices",
                "qualitative_panels"
            ],
        },
        "implementation_readiness": {
            "capture_supported_change_families": ["open", "toggle", "dirty", "fill"],
            "capture_pending": [
                "move_receptacle", "cook", "slice", "break", "multi_object_actions",
                "translated_camera", "rotated_camera", "partial_occlusion", "procthor"
            ],
            "language_parser_experiment": "adapter_implemented_exact_models_and_capture_pending",
            "inference_and_evaluation_pipeline": "implemented",
            "real_video_manifest_and_preparer": "implemented",
            "real_or_public_media": "not_yet_collected",
            "captured_real_vlm_responses": "not_yet_available",
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    validate_visual_experiment_protocol(payload)
    return payload


def validate_visual_experiment_protocol(payload: dict[str, object]) -> None:
    if payload.get("protocol_id") != VISUAL_EXPERIMENT_PROTOCOL:
        raise ValueError("visual experiment protocol ID is invalid")
    systems = payload.get("systems")
    comparisons = payload.get("primary_comparisons")
    tiers = payload.get("evidence_tiers")
    if not isinstance(systems, list) or not isinstance(comparisons, list) or not isinstance(tiers, list):
        raise ValueError("visual experiment protocol collections are malformed")
    system_ids = [row.get("id") for row in systems if isinstance(row, dict)]
    if len(system_ids) != len(systems) or len(system_ids) != len(set(system_ids)):
        raise ValueError("visual experiment system IDs must be unique")
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("visual experiment comparison is malformed")
        if comparison.get("system") not in system_ids or comparison.get("baseline") not in system_ids:
            raise ValueError("visual experiment comparison references an unknown system")
    tier_ids = [row.get("id") for row in tiers if isinstance(row, dict)]
    if len(tier_ids) != len(tiers) or len(tier_ids) != len(set(tier_ids)):
        raise ValueError("visual evidence tier IDs must be unique")
    raw_hash = payload.get("protocol_sha256")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        raise ValueError("visual experiment protocol hash is invalid")
    unhashed = dict(payload)
    del unhashed["protocol_sha256"]
    canonical = json.dumps(unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if raw_hash != expected:
        raise ValueError("visual experiment protocol content hash drifted")


def write_visual_experiment_protocol(path: str | Path, *, check: bool = False) -> dict[str, object]:
    destination = Path(path)
    payload = build_visual_experiment_protocol()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if check:
        if not destination.is_file():
            raise FileNotFoundError(f"missing frozen visual experiment protocol: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise ValueError("frozen visual experiment protocol drifted")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return payload
