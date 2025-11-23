from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import base64
import io
import insightface
from nudenet import NudeDetector

app = FastAPI(title="Self-Hosted Vision Server")

# ------------------------------
# Load Models
# ------------------------------

# InsightFace: SCRFD + MobileFaceNet
face_model = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_model.prepare(ctx_id=-1)

# NudeNet v3 detector
detector = NudeDetector()

# ------------------------------
# Request Types
# ------------------------------

class VerifyRequest(BaseModel):
    image1: str  # base64 string
    image2: str  # base64 string

class NSFWRequest(BaseModel):
    image: str  # base64

class FaceDetectRequest(BaseModel):
    image: str  # base64

# ------------------------------
# Helpers
# ------------------------------

def image_from_base64(b64: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64)
        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

# ------------------------------
# Face Verification
# ------------------------------

@app.post("/verify")
def verify_faces(req: VerifyRequest):
    try:
        img1 = image_from_base64(req.image1)
        img2 = image_from_base64(req.image2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    faces1 = face_model.get(img1)
    faces2 = face_model.get(img2)

    if not faces1 or not faces2:
        raise HTTPException(status_code=400, detail="No face detected in one or both images")

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    return {
        "similarity": similarity,
        "match": similarity > 0.40  # Suggested threshold
    }

# ------------------------------
# Face Detection API
# ------------------------------

@app.post("/detect_face")
def detect_face(req: FaceDetectRequest):
    try:
        img = image_from_base64(req.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    faces = face_model.get(img)

    return {
        "faces": len(faces),
        "hasFace": len(faces) > 0,
        "bboxes": [f.bbox.tolist() for f in faces]
    }

# ------------------------------
# NSFW Detection API
# ------------------------------

# Sexual exposure labels to compute "unsafe" score
sexual_labels = [
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
]

@app.post("/nsfw")
def nsfw_check(req: NSFWRequest):
    try:
        img = image_from_base64(req.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    detections = detector.detect(img)
    
    # Unsafe score: sum of sexual exposure probabilities
    unsafe_score = sum(d["score"] for d in detections if d["class"] in sexual_labels)

    return {
        "detections": detections,
        "unsafe_score": unsafe_score,
        "unsafe": unsafe_score > 0.3
    }
