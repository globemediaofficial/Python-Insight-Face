from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import insightface
import numpy as np
from PIL import Image
import base64
import io

app = FastAPI(title="Face Verification Server")

# Initialize InsightFace model (CPU mode)
model = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
model.prepare(ctx_id=-1, nms=0)  # CPU

class VerifyRequest(BaseModel):
    image1: str  # base64
    image2: str  # base64

def image_from_base64(b64: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64)
        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

@app.post("/verify")
def verify_faces(req: VerifyRequest):
    try:
        img1 = image_from_base64(req.image1)
        img2 = image_from_base64(req.image2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    faces1 = model.get(img1)
    faces2 = model.get(img2)

    if not faces1 or not faces2:
        raise HTTPException(status_code=400, detail="No face detected in one or both images")

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    return {"similarity": similarity, "match": similarity > 0.5}
