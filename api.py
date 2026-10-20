"""FastAPI service for frozen V4. No training, downloads or email persistence."""
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from antispam import SpamFilter
from service import MAX_REQUEST_BYTES, predict_payload


@asynccontextmanager
async def lifespan(app):
    app.state.model = SpamFilter()
    yield


app = FastAPI(title="NLP anti-spam V4", version="4.0", lifespan=lifespan)


@app.get("/health")
def health():
    return app.state.model.info()


@app.post("/predict")
async def predict(request: Request):
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/json":
        return JSONResponse({"error": "Use application/json"}, status_code=415)
    if request.headers.get("transfer-encoding"):
        return JSONResponse({"error": "Chunked requests are unsupported"}, status_code=400)
    try:
        length = int(request.headers.get("content-length", "-1"))
    except ValueError:
        return JSONResponse({"error": "Invalid Content-Length"}, status_code=400)
    if length < 0:
        return JSONResponse({"error": "Content-Length required"}, status_code=411)
    if length > MAX_REQUEST_BYTES:
        return JSONResponse({"error": "Request exceeds size limit"}, status_code=413)
    try:
        raw = await request.body()
        if len(raw) != length or len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Invalid body length")
        payload = json.loads(raw.decode("utf-8"))
        return await run_in_threadpool(predict_payload, app.state.model, payload)
    except (ValueError, UnicodeError):
        return JSONResponse({"error": "Invalid message or JSON payload"}, status_code=400)
