import io
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------
# SimpleCNN Model
# ------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ------------------
# Trainer + SSE queue per session
# ------------------
@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3

class Trainer:
    def __init__(self, session_key: str, device: Optional[str] = None):
        self.session_key = session_key
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=2048)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._last_metrics = {}

    def reset(self):
        self.stop()
        import queue as _q
        self._queue = _q.Queue(maxsize=2048)
        self._stop.clear()
        self._thread = None
        self._last_metrics = {}
        # สร้างโมเดล/ออปติไมเซอร์ใหม่
        self.model = SimpleCNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _put_html(self, html: str):
# Each SSE data payload is an HTML snippet with OOB swap to append to #log
        payload = html.replace("\n", " ")
        print(f"[DEBUG] _put_html: {payload}")
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            print("[DEBUG] _put_html: queue full, message dropped")
            pass
    def start(self, cfg: TrainConfig):
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._put_html(f"<li hx-swap-oob=\"beforeend:#log\">⚠️ Training already running…</li>")
                return False

            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
            self._thread.start()
            self._put_html("<li hx-swap-oob=\"beforeend:#log\">🚀 Started training…</li>")
            return True    
    # ...existing code...
    def _run(self, cfg: TrainConfig):
        print(f"[DEBUG] _run: started with cfg={cfg}")
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            print(f"[DEBUG] _run: loaded MNIST dataset, len={len(train_ds)}")
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

            global_step = 0
            for epoch in range(1, cfg.epochs + 1):
                if self._stop.is_set():
                    print(f"[DEBUG] _run: stop event set, breaking epoch loop")
                    break
                epoch_loss = 0.0
                correct = 0
                total = 0

                self.model.train()
                for batch_idx, (images, labels) in enumerate(train_loader, start=1):
                    if self._stop.is_set():
                        print(f"[DEBUG] _run: stop event set, breaking batch loop")
                        break
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    global_step += 1
                    if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                        avg_loss = epoch_loss / total
                        acc = correct / total
                        self._last_metrics = {"epoch": epoch, "step": global_step, "loss": avg_loss, "acc": acc}
                        print(f"[DEBUG] _run: epoch={epoch}, batch={batch_idx}, loss={avg_loss:.4f}, acc={acc:.4f}")
                        self._put_html(
                            f"<li hx-swap-oob=\"beforeend:#log\">Epoch {epoch}/{cfg.epochs} · Batch {batch_idx}/{len(train_loader)} · "
                            f"loss={avg_loss:.4f} · acc={acc:.4f}</li>"
                        )

            self._put_html("<li hx-swap-oob=\"beforeend:#log\">✅ Training finished.</li>")
            print(f"[DEBUG] _run: finished training loop")
        except Exception as e:
            print(f"[DEBUG] _run: exception: {e}")
            self._put_html(f"<li hx-swap-oob=\"beforeend:#log\" style=\"color:#ff6b6b\">❌ Error: {e}</li>")
# ...existing code...
    def event_iter(self):
# SSE generator: yield OOB HTML snippets to append to #log
        yield "retry: 1000\n\n" # reconnection delay
        while not self._stop.is_set():
            try:
                html = self._queue.get(timeout=5)
                yield f"event: message\ndata: {html}\n\n"
            except queue.Empty:
                yield "event: ping\ndata: keepalive\n\n"

    def predict(self, image_tensor: torch.Tensor):
# image_tensor: (1,1,28,28), normalized roughly like training
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)
            top3_prob, top3_idx = torch.topk(probs, 3)
            return {
                "pred": int(top_idx.item()),
                "conf": float(top_prob.item()),
                "top3": [(int(i.item()), float(p.item())) for p, i in zip(top3_prob, top3_idx)],
            }

    def stop(self):
        self._stop.set()
_trainers: Dict[str, Trainer] = {}
def get_trainer(session_key: str) -> Trainer:
    t = _trainers.get(session_key)
    if t is None:
        t = Trainer(session_key)
        _trainers[session_key] = t
    return t
