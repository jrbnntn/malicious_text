from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from detector.predict import MODEL_PATH, SPAM_THRESHOLD, _load_model
from .schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = _load_model()
    yield


app = FastAPI(
    title="Malicious Text Detector API",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict_message(payload: PredictRequest, request: Request) -> PredictResponse:
    model = request.app.state.model
    spam_score = float(model.predict_proba([payload.text])[0, 1])
    label = "spam" if spam_score >= SPAM_THRESHOLD else "ham"
    return PredictResponse(label=label, score=spam_score)


WEB_DIRECTORY = Path(__file__).resolve().parents[1] / "web"
app.mount("/", StaticFiles(directory=WEB_DIRECTORY, html=True), name="web")
