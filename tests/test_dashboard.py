import base64
import hashlib
import json

import pytest

import offender
from config import camera_source, load_dotenv


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFENDERS_DATA_FILE", str(tmp_path / "offenders_data.json"))
    monkeypatch.setenv("OFFENDERS_STATIC_DIR", str(tmp_path / "static"))
    folder = tmp_path / "static" / "no_helmet" / "NO_HELMET_session_demo"
    folder.mkdir(parents=True)
    (folder / "snapshot_demo.jpg").write_bytes(b"not-a-real-jpeg")
    (tmp_path / "static" / "no_helmet" / "bad name").mkdir()
    (tmp_path / "static" / "triples").mkdir()
    offender.app.config["TESTING"] = True
    return offender.app.test_client()


def csrf_token(response):
    marker = b'name="csrf-token" content="'
    start = response.data.index(marker) + len(marker)
    end = response.data.index(b'"', start)
    return response.data[start:end].decode()


def test_dashboard_lists_safe_sessions_and_escapes_html(client):
    page = client.get("/")
    assert page.status_code == 200
    token = csrf_token(page)
    updated = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "NO_HELMET_session_demo",
            "field": "name",
            "value": '<script>alert("xss")</script>',
        },
        headers={"X-CSRF-Token": token},
    )
    assert updated.status_code == 200
    again = client.get("/")
    assert b"<script>alert" not in again.data
    assert b"&lt;script&gt;alert" in again.data
    assert b"bad name" not in again.data
    assert b"snapshot_demo.jpg" in again.data
    assert b"noindex, nofollow" in again.data
    assert b"application/ld+json" in again.data


def test_update_requires_csrf_and_same_origin(client):
    page = client.get("/")
    token = csrf_token(page)
    missing = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "NO_HELMET_session_demo",
            "field": "location",
            "value": "Sample junction",
        },
    )
    assert missing.status_code == 403
    foreign = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "NO_HELMET_session_demo",
            "field": "location",
            "value": "Sample junction",
        },
        headers={"X-CSRF-Token": token, "Origin": "https://evil.example"},
    )
    assert foreign.status_code == 403
    allowed = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "NO_HELMET_session_demo",
            "field": "fine",
            "value": "500",
        },
        headers={"X-CSRF-Token": token},
    )
    assert allowed.status_code == 200
    with open(offender.data_path(), encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["no_helmet"]["NO_HELMET_session_demo"]["fine"] == 500


def test_rejects_unexpected_fields_and_path_traversal(client):
    page = client.get("/")
    token = csrf_token(page)
    headers = {"X-CSRF-Token": token}
    for field in ("offence", "__proto__", "fine_applied_note"):
        response = client.post(
            "/update_offender",
            json={
                "category": "no_helmet",
                "folder": "NO_HELMET_session_demo",
                "field": field,
                "value": "changed",
            },
            headers=headers,
        )
        assert response.status_code == 400
    traversal = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "../offenders_data",
            "field": "name",
            "value": "nope",
        },
        headers=headers,
    )
    assert traversal.status_code == 400
    negative = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "NO_HELMET_session_demo",
            "field": "fine",
            "value": -5,
        },
        headers=headers,
    )
    assert negative.status_code == 400


def test_fine_applied_checkbox_uses_real_booleans(client):
    page = client.get("/")
    token = csrf_token(page)
    response = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "NO_HELMET_session_demo",
            "field": "fine_applied",
            "value": True,
        },
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 200
    page = client.get("/")
    assert b"fine-applied" in page.data
    string_true = client.post(
        "/update_offender",
        json={
            "category": "no_helmet",
            "folder": "NO_HELMET_session_demo",
            "field": "fine_applied",
            "value": "false",
        },
        headers={"X-CSRF-Token": token},
    )
    assert string_true.status_code == 400


def test_security_headers_vendor_files_and_discovery_routes(client):
    page = client.get("/")
    assert page.headers["X-Frame-Options"] == "DENY"
    assert "script-src" in page.headers["Content-Security-Policy"]
    assert page.headers["X-Robots-Tag"] == "noindex, nofollow"
    start = page.get_data(as_text=True).find('application/ld+json">') + len('application/ld+json">')
    end = page.get_data(as_text=True).find("</script>", start)
    blob = page.get_data(as_text=True)[start:end]
    digest = base64.b64encode(hashlib.sha256(blob.encode()).digest()).decode()
    assert digest in page.headers["Content-Security-Policy"]
    assert client.get("/vendor/bootstrap.min.css").status_code == 200
    assert client.get("/vendor/bootstrap.bundle.min.js").status_code == 200
    robots = client.get("/robots.txt")
    assert b"Disallow: /update_offender" in robots.data
    assert b"Sitemap:" in robots.data
    assert b"Helmet and Triple Riding Detection" in client.get("/llms.txt").data
    assert b"<loc>" in client.get("/sitemap.xml").data


def test_corrupt_data_file_does_not_crash_into_the_debugger(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OFFENDERS_DATA_FILE", str(tmp_path / "broken.json"))
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    response = client.get("/")
    assert response.status_code == 500
    assert b"unreadable" in response.data


def test_debugger_is_refused_off_loopback(monkeypatch):
    monkeypatch.setenv("FLASK_DEBUG", "1")
    monkeypatch.setenv("FLASK_HOST", "0.0.0.0")
    host, _port, debug = offender.server_options()
    assert host == "0.0.0.0"
    assert debug is False
    monkeypatch.setenv("FLASK_HOST", "127.0.0.1")
    assert offender.server_options()[2] is True


def test_camera_source_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE", "0")
    assert camera_source() == 0
    monkeypatch.setenv("CAMERA_SOURCE", "webcam")
    with pytest.raises(SystemExit):
        camera_source()
    env_file = tmp_path / ".env"
    env_file.write_text("CAMERA_SOURCE=4\nALREADY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ALREADY", "from-env")
    monkeypatch.delenv("CAMERA_SOURCE")
    load_dotenv(env_file)
    assert camera_source() == 4
    assert offender.os.environ["ALREADY"] == "from-env"
