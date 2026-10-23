import base64
import hashlib
import json
import os
import secrets

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, session

from config import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
    MAX_CONTENT_LENGTH=32 * 1024,
)

CATEGORIES = ("triples", "no_helmet")
ALLOWED_FIELDS = {"name", "number_plate", "fine", "location", "fine_applied"}
TEXT_FIELDS = {"name", "number_plate", "location"}
MAX_TEXT = 200
MAX_FINE = 99_999_999
REPO_URL = "https://github.com/ryaneditraj/Helmet-and-Triples-Detection-using-Yolov5-and-Yolov8"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

STRUCTURED_DATA = json.dumps(
    {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "Helmet and Triple Riding Detection",
        "applicationCategory": "SecurityApplication",
        "operatingSystem": "Linux, Windows, macOS",
        "description": (
            "Open-source computer-vision tool that detects motorcycle riders "
            "without helmets with YOLOv8 and triple riding with YOLOv5, then "
            "reviews timestamped snapshots on a local Flask dashboard."
        ),
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        "license": "https://www.gnu.org/licenses/agpl-3.0.html",
        "codeRepository": REPO_URL,
        "url": REPO_URL,
        "image": (
            "https://raw.githubusercontent.com/ryaneditraj/"
            "Helmet-and-Triples-Detection-using-Yolov5-and-Yolov8/main/"
            "docs/screenshots/dashboard.png"
        ),
        "author": {
            "@type": "Person",
            "name": "Ryan Editraj",
            "url": "https://github.com/ryaneditraj",
            "sameAs": [
                "https://github.com/ryaneditraj",
                "https://instagram.com/ryaneditraj",
            ],
        },
        "keywords": (
            "helmet detection, triple riding detection, YOLOv8, YOLOv5, "
            "motorcycle safety, traffic violation dashboard, computer vision, Flask"
        ),
    },
    indent=2,
).replace("<", "\\u003c")


def data_path():
    return os.environ.get("OFFENDERS_DATA_FILE", "offenders_data.json")


def static_dir():
    return os.environ.get("OFFENDERS_STATIC_DIR", "static")


def content_security_policy():
    digest = hashlib.sha256(STRUCTURED_DATA.encode("utf-8")).digest()
    script_hash = base64.b64encode(digest).decode("ascii")
    return (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self'; "
        f"script-src 'self' 'sha256-{script_hash}'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "connect-src 'self'"
    )


def server_options():
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = (os.environ.get("FLASK_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.environ.get("FLASK_PORT", "5000"))
    except ValueError:
        raise SystemExit("FLASK_PORT must be an integer.")
    if not 1 <= port <= 65535:
        raise SystemExit("FLASK_PORT is out of range.")
    if debug and host not in LOOPBACK_HOSTS:
        print("FLASK_DEBUG is ignored on a non-local host. The debugger can run code.")
        debug = False
    return host, port, debug


def is_safe_token(value):
    if not isinstance(value, str) or not value or len(value) > 200:
        return False
    return all(character.isalnum() or character in {"_", "-"} for character in value)


def is_safe_image(name):
    if not isinstance(name, str) or not name or len(name) > 200:
        return False
    if name.startswith(".") or ".." in name or "/" in name or "\\" in name:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if any(character not in allowed for character in name):
        return False
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))


def real_child(parent, *parts):
    root = os.path.realpath(parent)
    candidate = os.path.realpath(os.path.join(root, *parts))
    try:
        if os.path.commonpath([root, candidate]) != root:
            return None
    except ValueError:
        return None
    return candidate


def load_data():
    path = data_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Offender data must be a JSON object")
    return data


def save_data(data):
    path = os.path.abspath(data_path())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
    os.replace(temporary, path)


def as_text(value):
    return value if isinstance(value, str) else ""


def public_record(raw, category):
    raw = raw if isinstance(raw, dict) else {}
    try:
        fine = int(raw.get("fine", 0))
    except (TypeError, ValueError):
        fine = 0
    offence = raw.get("offence")
    if not isinstance(offence, str) or not offence.strip():
        offence = "Triple Riding" if category == "triples" else "No Helmet"
    applied = raw.get("fine_applied", False)
    return {
        "name": as_text(raw.get("name")),
        "offence": offence,
        "number_plate": as_text(raw.get("number_plate")),
        "fine": fine,
        "location": as_text(raw.get("location")),
        "fine_applied": applied if isinstance(applied, bool) else False,
    }


def get_offenders():
    data = load_data()
    updated = False
    offenders = {}
    root = static_dir()

    for category in CATEGORIES:
        category_path = real_child(root, category)
        if category_path is None:
            continue
        os.makedirs(category_path, exist_ok=True)
        folder_names = [
            folder
            for folder in os.listdir(category_path)
            if is_safe_token(folder) and os.path.isdir(os.path.join(category_path, folder))
        ]
        folder_names.sort()
        offenders[category] = []

        if category not in data or not isinstance(data.get(category), dict):
            data[category] = {}
            updated = True

        for folder in folder_names:
            folder_path = real_child(category_path, folder)
            if folder_path is None or not os.path.isdir(folder_path):
                continue
            if folder not in data[category] or not isinstance(data[category].get(folder), dict):
                data[category][folder] = {
                    "name": "",
                    "offence": "Triple Riding" if category == "triples" else "No Helmet",
                    "number_plate": "",
                    "fine": 0,
                    "location": "",
                    "fine_applied": False,
                }
                updated = True

            images = [
                image
                for image in os.listdir(folder_path)
                if is_safe_image(image) and os.path.isfile(os.path.join(folder_path, image))
            ]
            images.sort()
            offenders[category].append(
                {
                    "folder": folder,
                    "images": images,
                    **public_record(data[category][folder], category),
                }
            )

    if updated:
        save_data(data)
    return offenders


def current_csrf():
    token = session.get("csrf_token")
    if not isinstance(token, str) or len(token) < 16:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def request_origin_allowed():
    origin = request.origin
    if not origin:
        return True
    return origin.rstrip("/") == request.host_url.rstrip("/")


@app.before_request
def guard_post():
    if request.method != "POST":
        return None
    if not request_origin_allowed():
        return jsonify({"status": "error", "message": "Cross-origin request blocked"}), 403
    expected = session.get("csrf_token")
    sent = request.headers.get("X-CSRF-Token", "")
    if not isinstance(expected, str) or not sent or not secrets.compare_digest(sent, expected):
        return jsonify({"status": "error", "message": "Invalid CSRF token"}), 403
    return None


@app.after_request
def harden(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = content_security_policy()
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    if request.path == "/":
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Cache-Control"] = "no-store"
    if request.path.startswith("/static/no_helmet/") or request.path.startswith("/static/triples/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    try:
        offenders = get_offenders()
    except (json.JSONDecodeError, ValueError, OSError):
        return Response("Offender data file is unreadable.", status=500, mimetype="text/plain")
    return render_template(
        "index.html",
        offenders=offenders,
        csrf_token=current_csrf(),
        structured_data=STRUCTURED_DATA,
        repo_url=REPO_URL,
    )


@app.route("/update_offender", methods=["POST"])
def update_offender():
    content = request.get_json(silent=True)
    if not isinstance(content, dict):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    category = content.get("category")
    folder = content.get("folder")
    field = content.get("field")
    value = content.get("value")

    if category not in CATEGORIES or not is_safe_token(folder) or field not in ALLOWED_FIELDS:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    if field in TEXT_FIELDS:
        if not isinstance(value, str):
            return jsonify({"status": "error", "message": "Text fields must be strings"}), 400
        value = value.strip()
        if len(value) > MAX_TEXT:
            return jsonify({"status": "error", "message": "Value is too long"}), 400
    elif field == "fine":
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return jsonify({"status": "error", "message": "Fine must be a number"}), 400
        try:
            value = int(value)
        except ValueError:
            return jsonify({"status": "error", "message": "Fine must be a number"}), 400
        if value < 0 or value > MAX_FINE:
            return jsonify({"status": "error", "message": "Fine must be a number"}), 400
    elif field == "fine_applied":
        if not isinstance(value, bool):
            return jsonify({"status": "error", "message": "Fine applied must be true or false"}), 400

    try:
        data = load_data()
    except (json.JSONDecodeError, ValueError, OSError):
        return jsonify({"status": "error", "message": "Offender data file is unreadable"}), 500

    if category not in data or not isinstance(data.get(category), dict) or folder not in data[category]:
        return jsonify({"status": "error", "message": "Offender not found"}), 404
    if not isinstance(data[category][folder], dict):
        return jsonify({"status": "error", "message": "Offender not found"}), 404

    data[category][folder][field] = value
    save_data(data)
    return jsonify({"status": "success"})


@app.route("/vendor/bootstrap.min.css")
def vendor_css():
    return send_from_directory(os.path.join(app.root_path, "templates", "css"), "bootstrap.min.css")


@app.route("/vendor/bootstrap.bundle.min.js")
def vendor_js():
    return send_from_directory(os.path.join(app.root_path, "templates", "js"), "bootstrap.bundle.min.js")


@app.route("/robots.txt")
def robots():
    body = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /update_offender\n"
        "Disallow: /static/no_helmet/\n"
        "Disallow: /static/triples/\n"
        "\n"
        f"Sitemap: {request.host_url}sitemap.xml\n"
    )
    return Response(body, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "  <url>\n"
        f"    <loc>{request.host_url}</loc>\n"
        "    <changefreq>weekly</changefreq>\n"
        "    <priority>1.0</priority>\n"
        "  </url>\n"
        "</urlset>\n"
    )
    return Response(xml, mimetype="application/xml")


@app.route("/llms.txt")
def llms():
    return send_from_directory(app.root_path, "llms.txt", mimetype="text/plain")


if __name__ == "__main__":
    host, port, debug = server_options()
    if host not in LOOPBACK_HOSTS:
        print("Warning: this dashboard has no login. Do not expose it on an untrusted network.")
    app.run(host=host, port=port, debug=debug)
