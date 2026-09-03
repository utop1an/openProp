from __future__ import annotations

import json
import os
import platform
import subprocess
from importlib.metadata import version


def command(arguments: list[str]) -> dict[str, object]:
    result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    return {
        "command": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> None:
    report: dict[str, object] = {
        "host": platform.node(),
        "python": platform.python_version(),
        "ai2thor": version("ai2thor"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": command(["nvidia-smi", "-L"]),
        "vulkan": command(["vulkaninfo", "--summary"]),
    }
    try:
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering

        controller = Controller(
            scene="FloorPlan1", platform=CloudRendering, width=320, height=240,
            renderInstanceSegmentation=True,
        )
        event = controller.step(action="RotateRight")
        report["controller"] = {
            "created": True,
            "last_action_success": event.metadata.get("lastActionSuccess"),
            "frame_shape": list(event.frame.shape),
            "instance_boxes": len(event.instance_detections2D),
        }
        controller.stop()
    except Exception as error:
        report["controller"] = {
            "created": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
