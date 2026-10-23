# Contributing

Thanks for helping improve helmet and triple-riding detection. This project is licensed under the GNU AGPL-3.0. Contributions are licensed the same way.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

`requirements-dev.txt` is enough for the dashboard and geometry tests. Install `requirements.txt` when you want to run the camera scripts. Those scripts need a webcam and, the first time `triples.py` runs, a download of the YOLOv5 source through `torch.hub`.

Copy `.env.example` to `.env` if you want to override the host, port, or camera index.

## Pull requests

1. Fork the repository and branch from `main`.
2. Keep the detection rules and folder names working. Snapshot folders use `NO_HELMET_session_*` and `THRIPLES_session_*`. The triple-riding prefix is spelled that way on purpose so older folders still match.
3. Run `pytest` before you open the pull request.
4. Describe what changed and how you checked it.
5. Do not commit `.env`, webcam snapshots, or new checkpoints you cannot explain. Snapshots can contain faces and plate numbers.

## Reporting issues

Use the bug and feature templates. Security reports go through a private advisory, as described in [SECURITY.md](SECURITY.md). Conduct reports are covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
