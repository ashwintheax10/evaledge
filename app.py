from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import base64
from PIL import Image
from insightface.app import FaceAnalysis
import onnxruntime as ort

app = Flask(__name__)

# ---- CONFIGURATION ----
UPLOAD_FOLDER = 'reference'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- AUTO-DETECT EXECUTION PROVIDER ----
providers = ort.get_available_providers()
print(f"🧠 Available ONNX providers: {providers}")

if 'CUDAExecutionProvider' in providers:
    chosen_provider = ['CUDAExecutionProvider']
    print("✅ Using GPU via CUDAExecutionProvider")
else:
    chosen_provider = ['CPUExecutionProvider']
    print("⚠️ GPU not available, using CPUExecutionProvider")

# ---- INITIALIZE FACE ANALYSIS ----
face_app = FaceAnalysis(providers=chosen_provider)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ---- IMAGE UTILS ----
def load_image(path):
    try:
        pil_image = Image.open(path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Error loading image '{path}': {e}")
        return None

def get_embedding(image):
    faces = face_app.get(image)
    if not faces:
        print("❌ No face detected in the image.")
        return None
    embedding = faces[0].embedding
    return embedding / np.linalg.norm(embedding)

# ---- LOAD REFERENCE IMAGES ----
reference_embeddings = {}
print("📦 Loading reference images...")
for filename in os.listdir(UPLOAD_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"➡️ Processing: {path}")
        img = load_image(path)
        if img is None:
            print(f"❌ Failed to load: {filename}")
            continue
        emb = get_embedding(img)
        if emb is None:
            print(f"❌ No face found in: {filename}")
            continue
        reference_embeddings[filename] = emb
        print(f"✅ Loaded and embedded: {filename}")
print("✅ Reference images ready:", list(reference_embeddings.keys()))

# ---- ROUTES ----
@app.route('/')
def index():
    return render_template('capture.html')

@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.form['image']
        encoded = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        live_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print("✅ Live image received and decoded.")

        live_emb = get_embedding(live_image)
        if live_emb is None:
            cv2.imwrite("debug_no_face.jpg", live_image)  # Optional: for debugging
            return "❌ No face detected in the captured image."

        print(f"📐 Live embedding norm: {np.linalg.norm(live_emb):.4f}")

        max_score = 0
        matched_person = "Unknown"
        threshold = 0.6

        for name, ref_emb in reference_embeddings.items():
            score = np.dot(ref_emb, live_emb)
            print(f"🧪 Comparing with {name} — Score: {score:.4f}")
            if score > max_score:
                max_score = score
                matched_person = name

        if max_score >= threshold:
            return f"✅ Match found: {matched_person} (Similarity: {max_score:.4f})"
        else:
            return f"❌ No match found. Closest: {matched_person} (Score: {max_score:.4f})"

    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        return f"❌ Error during verification: {str(e)}"

# ---- RUN SERVER ----
if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True)
