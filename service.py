"""Shared request validation and local FastAPI/Uvicorn entry point."""
import base64
import binascii

MAX_REQUEST_BYTES = 2_800_000


def predict_payload(model, payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON request must be an object")
    if "eml_base64" in payload:
        if set(payload) != {"eml_base64"} or not isinstance(payload["eml_base64"], str):
            raise ValueError("eml_base64 must be the only field and a string")
        try:
            raw = base64.b64decode(payload["eml_base64"], validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Invalid eml_base64")
        return model.predict_eml(raw)
    if not set(payload) <= {"subject", "body", "html"} or not ({"subject", "body"} & set(payload)):
        raise ValueError("Expected subject/body and optional html, or eml_base64")
    return model.predict_text(payload.get("subject", ""), payload.get("body", ""), payload.get("html", False))


def serve(host, port):
    import uvicorn
    from api import app
    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)
