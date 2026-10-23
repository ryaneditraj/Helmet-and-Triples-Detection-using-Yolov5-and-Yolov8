import os


def load_dotenv(path=None):
    """Load KEY=VALUE pairs from a local env file without overriding the process environment."""
    path = path or os.environ.get("ENV_FILE", ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("'").strip('"')


def camera_source(prompt="ENTER SOURCE (0 for OBS, and 1 for webcam)"):
    """Read a webcam index from CAMERA_SOURCE or from the same interactive prompt as before."""
    raw = os.environ.get("CAMERA_SOURCE")
    if raw is None or raw.strip() == "":
        raw = input(prompt)
    raw = str(raw).strip()
    try:
        source = int(raw)
    except ValueError:
        print("Camera source must be an integer device index.")
        raise SystemExit(2)
    if source < 0:
        print("Camera source must be zero or a positive device index.")
        raise SystemExit(2)
    return source
