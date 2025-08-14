import os
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATA_FILE = "offenders_data.json"
STATIC_DIR = "static"
CATEGORIES = ["triples", "no_helmet"]

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def get_offenders():
    data = load_data()
    updated = False
    offenders = {}

    for category in CATEGORIES:
        cat_path = os.path.join(STATIC_DIR, category)
        os.makedirs(cat_path, exist_ok=True)

        folder_names = [f for f in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, f))]
        offenders[category] = []

        if category not in data:
            data[category] = {}

        for folder in folder_names:
            if folder not in data[category]:
                data[category][folder] = {
                    "name": "",
                    "offence": "Triple Riding" if category == "triples" else "No Helmet",
                    "number_plate": "",
                    "fine": 0,
                    "location": "",
                    "fine_applied": False
                }
                updated = True

            image_dir = os.path.join(cat_path, folder)
            images = [img for img in os.listdir(image_dir)
                      if img.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

            offenders[category].append({
                "folder": folder,
                "images": images,
                **data[category][folder]
            })

    if updated:
        save_data(data)

    return offenders

@app.route("/")
def index():
    offenders = get_offenders()
    return render_template("index.html", offenders=offenders)

@app.route("/update_offender", methods=["POST"])
def update_offender():
    data = load_data()
    content = request.get_json()

    category = content.get("category")
    folder = content.get("folder")
    field = content.get("field")
    value = content.get("value")

    if not all([category, folder, field]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    if category not in data or folder not in data[category]:
        return jsonify({"status": "error", "message": "Offender not found"}), 404

    # Type casting
    if field == "fine":
        try:
            value = int(value)
        except ValueError:
            return jsonify({"status": "error", "message": "Fine must be a number"}), 400
    elif field == "fine_applied":
        value = bool(value)

    data[category][folder][field] = value
    save_data(data)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
