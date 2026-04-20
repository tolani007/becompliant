"""Run the Roboflow workflow on a single image and dump the raw JSON.

Use this to verify the exact keys returned by the workflow before wiring the UI.

Usage:
    python sanity_check.py path/to/cooler.jpg
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python sanity_check.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    client = InferenceHTTPClient(
        api_url=os.environ["ROBOFLOW_API_URL"],
        api_key=os.environ["ROBOFLOW_API_KEY"],
    )

    result = client.run_workflow(
        workspace_name=os.environ["ROBOFLOW_WORKSPACE"],
        workflow_id=os.environ["ROBOFLOW_WORKFLOW_ID"],
        images={"image": str(image_path)},
    )

    # Strip base64 image blobs so the console dump stays readable.
    def redact(value):
        if isinstance(value, str) and len(value) > 500:
            return f"<{len(value)}-char string, likely base64>"
        if isinstance(value, dict):
            return {k: redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    print(json.dumps(redact(result), indent=2, default=str))


if __name__ == "__main__":
    main()
