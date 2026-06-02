import torch
import timm
from torchvision import transforms
from PIL import Image
import gradio as gr
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# LOAD MODEL
# ===============================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=3
)

model.load_state_dict(
    torch.load("WasteIQ_EfficientNetLast.pth", map_location=device)
)

model.to(device)
model.eval()

class_names = ["metal", "paper", "plastic"]

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===============================
# FILE PATHS
# ===============================
LATEST_IMAGE_PATH = "latest.jpg"
HISTORY_FILE = "history.json"

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# ===============================
# RUN MODEL
# ===============================
def run_model(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    prediction_dict = {
        class_names[i]: float(probs[i])
        for i in range(3)
    }

    predicted_index = probs.argmax()
    predicted_class = class_names[predicted_index]
    confidence = float(probs[predicted_index])  # ⭐ FINAL CONFIDENCE

    return prediction_dict, predicted_class, confidence

# ===============================
# SAVE HISTORY
# ===============================
def save_history(predicted_class, probabilities):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    history.append({
        "prediction": predicted_class,
        "probabilities": probabilities,
        "time": str(datetime.now())
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# ===============================
# GENERATE BAR GRAPH (UPDATED)
# ===============================
def generate_graph():
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    counts = {"metal": 0, "paper": 0, "plastic": 0}

    for entry in history:
        counts[entry["prediction"]] += 1

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(6, 4))

    bars = plt.barh(labels, values)

    # Add labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                 str(width), va='center')

    plt.xlabel("Count")
    plt.title("Waste Classification Distribution")

    plt.tight_layout()

    graph_path = "graph.png"
    plt.savefig(graph_path)
    plt.close()

    return graph_path

# ===============================
# FASTAPI APP
# ===============================
app = FastAPI()

@app.post("/predict")
async def predict_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        image.save(LATEST_IMAGE_PATH)

        prediction_dict, predicted_class, confidence = run_model(image)

        save_history(predicted_class, prediction_dict)

        return {
            "status": "success",
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),  # %
            "probabilities": prediction_dict
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ===============================
# DASHBOARD DATA
# ===============================
def get_dashboard_data():
    image = Image.open(LATEST_IMAGE_PATH) if os.path.exists(LATEST_IMAGE_PATH) else None
    graph = generate_graph()

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    if history:
        last_entry = history[-1]
        last_prediction = last_entry["prediction"]
        probabilities = last_entry["probabilities"]

        confidence = round(max(probabilities.values()) * 100, 2)

        total_items = len(history)
    else:
        last_prediction = "No Data"
        probabilities = {}
        confidence = 0
        total_items = 0

    return image, graph, last_prediction, probabilities, total_items, confidence

# ===============================
# GRADIO DASHBOARD UI
# ===============================
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    body { background-color: #0E1117; color: white; }
    .gr-button {
        background-color: #1F6FEB !important;
        color: white !important;
        font-weight: bold;
    }
    """
) as demo:

    gr.Markdown("## ♻️ WasteIQ – Industrial Smart Waste Dashboard")

    with gr.Row():
        total_box = gr.Number(label="Total Classified Waste")
        last_pred_box = gr.Textbox(label="Last Prediction")
        confidence_box = gr.Textbox(label="Confidence (%)")

    with gr.Row():
        live_image = gr.Image(label="📷 Latest ESP Capture")
        graph_output = gr.Image(label="📊 Waste Distribution (Bar Chart)")

    prob_table = gr.JSON(label="📈 Prediction Probabilities")

    refresh_btn = gr.Button("🔄 Refresh Dashboard")

    refresh_btn.click(
        fn=get_dashboard_data,
        outputs=[
            live_image,
            graph_output,
            last_pred_box,
            prob_table,
            total_box,
            confidence_box
        ],
        show_progress=True
    )

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")
