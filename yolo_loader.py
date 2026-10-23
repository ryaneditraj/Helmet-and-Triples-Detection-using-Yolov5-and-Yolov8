import os

import torch


def load_yolov5(weights_path=None):
    """Load the committed YOLOv5 weights.

    torch.hub clones github.com/ultralytics/yolov5 and runs that code.
    trust_repo=True skips the interactive prompt so the camera script can start
    unattended. Only keep using it if you trust that repository. See SECURITY.md.
    """
    weights_path = weights_path or os.environ.get("TRIPLE_WEIGHTS", "yolov5s.pt")
    if os.path.isfile(weights_path):
        return torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=weights_path,
            trust_repo=True,
        )
    return torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True,
        trust_repo=True,
    )
