from django.shortcuts import render

import base64
import io
from PIL import Image, ImageOps

from django.http import StreamingHttpResponse, HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.cache import never_cache

from .trainer import get_trainer, TrainConfig


def _ensure_session(request):
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


@ensure_csrf_cookie
@require_http_methods(["GET"])
def training_page(request):
    _ensure_session(request)
    return render(request, "training.html")

@require_http_methods(["POST"])
def start_training(request):
    session_key = _ensure_session(request)
    trainer = get_trainer(session_key)

# Read optional params (epochs, lr, batch_size) from POST
    try:
        epochs = int(request.POST.get("epochs", 2))
        batch_size = int(request.POST.get("batch_size", 64))
        lr = float(request.POST.get("lr", 1e-3))
    except ValueError:
        return HttpResponseBadRequest("Invalid training parameters")

    trainer.start(TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr))

# Update status area via htmx target
    return HttpResponse(f"Running (epochs={epochs}, batch={batch_size}, lr={lr})")


@never_cache
@require_http_methods(["GET"])
def train_stream(request):
    session_key = _ensure_session(request)
    trainer = get_trainer(session_key)

    resp = StreamingHttpResponse(trainer.event_iter(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no" # helpful behind proxies like nginx
    return resp


@ensure_csrf_cookie
@require_http_methods(["GET"])
def predict_page(request):
    _ensure_session(request)
    return render(request, "predict.html")

@require_http_methods(["POST"])
def predict_digit(request):
    session_key = _ensure_session(request)
    trainer = get_trainer(session_key)

    data_url = request.POST.get("image_data")
    if not data_url or not data_url.startswith("data:image"):
        return HttpResponseBadRequest("Missing or invalid image_data")

# Decode base64 PNG from data URL
    try:
        header,b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("L") # grayscale
# Resize to 28x28 and invert (canvas is typically white background, MNIST is white-on-black)
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
# Normalize approximately like training (0.1307 mean, 0.3081 std)
        import numpy as np
        arr = np.array(img, dtype="float32") / 255.0
        arr = (arr - 0.1307) / 0.3081
        import torch
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) # (1,1,28,28)
        out = trainer.predict(tensor)
    except Exception as e:
        return HttpResponseBadRequest(f"Failed to process image: {e}")

# Return HTML snippet (for htmx swap)
    top3_html = "".join([f"<li>{d} : {p:.2%}</li>" for d, p in out["top3"]])
    html = f"""
<div>
<div class=\"pred\">Prediction: <strong>{out['pred']}</strong></div>
<div class=\"conf\">Confidence: {out['conf']:.2%}</div>
<ul class=\"top3\">{top3_html}</ul>
</div>
"""
    return HttpResponse(html)