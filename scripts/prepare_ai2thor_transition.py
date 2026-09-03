from __future__ import annotations

import argparse
import json
from pathlib import Path

from openprop.ai2thor_adapter import (
    AI2ThorFrameBundle,
    AI2ThorTransitionTruth,
    derive_ai2thor_transition,
    extract_ai2thor_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one AI2-THOR before/after transition while keeping "
            "VLM-visible inputs separate from evaluation-only simulator truth."
        )
    )
    parser.add_argument("--before-metadata", type=Path, required=True)
    parser.add_argument("--after-metadata", type=Path, required=True)
    parser.add_argument("--before-image", type=Path, required=True)
    parser.add_argument("--after-image", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--before-boxes", type=Path)
    parser.add_argument("--after-boxes", type=Path)
    parser.add_argument("--before-time", type=float, default=0.0)
    parser.add_argument("--after-time", type=float, default=1.0)
    parser.add_argument("--source", default="ai2thor-rgb")
    parser.add_argument("--movement-threshold-metres", type=float, default=0.05)
    return parser.parse_args()


def frame_input(bundle: AI2ThorFrameBundle) -> dict[str, object]:
    frame = bundle.frame
    return {
        "frame_id": frame.frame_id,
        "image_url": frame.image_url,
        "captured_at": frame.captured_at,
        "source": frame.source,
        "candidate_entity_ids": list(frame.candidate_entity_ids),
        "candidate_regions": {
            entity_id: list(region)
            for entity_id, region in frame.candidate_regions.items()
        },
    }


def truth_frame(bundle: AI2ThorFrameBundle) -> dict[str, object]:
    return {
        "frame_id": bundle.frame.frame_id,
        "scene_name": bundle.scene_name,
        "objects": [
            {
                "entity_id": item.entity_id,
                "object_type": item.object_type,
                "visible": item.visible,
                "values": dict(item.values),
            }
            for item in bundle.current_truth
        ],
    }


def transition_payload(
    transition: AI2ThorTransitionTruth,
) -> dict[str, object]:
    return {
        "scene_name": transition.scene_name,
        "action": transition.action,
        "before_frame_id": transition.before_frame_id,
        "after_frame_id": transition.after_frame_id,
        "changes": {
            entity_id: [
                {
                    "property_name": change.property_name,
                    "before": change.before,
                    "after": change.after,
                }
                for change in changes
            ]
            for entity_id, changes in transition.changes.items()
        },
    }


def main() -> None:
    args = parse_args()
    for image_path in (args.before_image, args.after_image):
        if not image_path.is_file():
            raise FileNotFoundError(f"missing captured image: {image_path}")
    before_metadata = json.loads(args.before_metadata.read_text(encoding="utf-8"))
    after_metadata = json.loads(args.after_metadata.read_text(encoding="utf-8"))
    before_boxes = (
        json.loads(args.before_boxes.read_text(encoding="utf-8"))
        if args.before_boxes is not None
        else None
    )
    after_boxes = (
        json.loads(args.after_boxes.read_text(encoding="utf-8"))
        if args.after_boxes is not None
        else None
    )
    before = extract_ai2thor_frame(
        before_metadata,
        frame_id="before",
        image_url=str(args.before_image.resolve()),
        captured_at=args.before_time,
        source=args.source,
        instance_detections_2d=before_boxes,
    )
    after = extract_ai2thor_frame(
        after_metadata,
        frame_id="after",
        image_url=str(args.after_image.resolve()),
        captured_at=args.after_time,
        source=args.source,
        instance_detections_2d=after_boxes,
    )
    transition = derive_ai2thor_transition(
        before,
        after,
        action=args.action,
        movement_threshold_metres=args.movement_threshold_metres,
    )

    input_path = args.output_prefix.with_suffix(".inputs.json")
    truth_path = args.output_prefix.with_suffix(".truth.json")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = {
        "schema_version": 1,
        "frames": [frame_input(before), frame_input(after)],
        "truth_artifact": truth_path.name,
    }
    truth = {
        "schema_version": 1,
        "frames": [truth_frame(before), truth_frame(after)],
        "transition": transition_payload(transition),
    }
    input_path.write_text(
        json.dumps(inputs, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    truth_path.write_text(
        json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "inputs": str(input_path),
                "truth": str(truth_path),
                "changed_entities": list(transition.changed_entity_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

