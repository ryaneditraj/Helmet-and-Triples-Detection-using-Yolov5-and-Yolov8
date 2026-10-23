# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security report.

Use a private GitHub security advisory:

https://github.com/ryaneditraj/Helmet-and-Triples-Detection-using-Yolov5-and-Yolov8/security/advisories/new

Include what you found, how to reproduce it, and the impact you expect. The maintainer is Ryan Editraj (https://github.com/ryaneditraj).

## Supported surface

Security fixes in this repository apply to the current `main` branch. There are no versioned releases yet.

## What was hardened

- The dashboard debugger is off unless `FLASK_DEBUG=1`, and it is refused when `FLASK_HOST` is not loopback. The Werkzeug debugger can execute Python on the machine.
- `POST /update_offender` only accepts `name`, `number_plate`, `fine`, `location`, and `fine_applied`. Category and folder must be a single safe path token. A CSRF token and a matching `Origin` are required.
- Pages send `Content-Security-Policy`, `X-Frame-Options: DENY`, `nosniff`, and `noindex`. Bootstrap is served from this repository instead of a CDN. Dashboard JavaScript is an external file, so the policy does not need `unsafe-inline`.
- Capture folders with spaces or path characters are not rendered. Snapshot responses are `Cache-Control: no-store`.
- Offender JSON is written to a temporary file and then replaced.
- Camera indexes must be integers. `CAMERA_SOURCE` is optional.
- Direct dependencies are pinned in `requirements.txt`, including Flask 3.1.3, Werkzeug 3.1.9, Jinja2 3.1.6, Pillow 13.0.0, urllib3 2.8.0, and PyTorch 2.14.1 (the 2.6.0 line fixed CVE-2025-32434). YOLOv5's hub code needs pandas, seaborn, and scipy, so those are pinned too.

## Remaining risk

- The dashboard has **no accounts**. Anyone who can open the port can view and edit records. Leave `FLASK_HOST` at `127.0.0.1`. Do not port-forward it.
- `triples.py` and `test.py` call `torch.hub` with `trust_repo=True`. That clones `github.com/ultralytics/yolov5` and executes it. Review that checkout before use, or run without a network after the hub cache exists.
- `.pt` checkpoints are PyTorch pickles. Load only the files shipped here, or files you trained. Do not load a checkpoint from a stranger.
- `offenders_data.json` and the snapshot folders can hold names, places, and plate numbers if someone types them in. Runtime capture folders are gitignored. Do not commit new records that contain that information.
- There is no rate limit on the update route.
- `test.py` is a development copy of the triple-riding loop. It writes `session_*` folders in the working directory instead of `static/triples/`.

## Secrets

No API keys, passwords, or private tokens were found in the tree or in the three commits on `main`. Do not add them. Put local settings in `.env`, using `.env.example` as the list of names.
