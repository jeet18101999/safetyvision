"""Training subprocess: fine-tunes one model on its accumulated dataset and evaluates new vs. current weights
on the same validation set. Run by LearningHub (separate process so the UI and live detection keep running)."""
import argparse
import json
from pathlib import Path


def evaluate(task, weights, data, imgsz, device):
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(data), split="val", imgsz=imgsz, device=device, workers=0,
                               plots=False, verbose=False, batch=16)
    if task == "classify":
        return {"score": float(m.top1), "top1": float(m.top1)}
    return {"score": float(m.box.map50), "map50": float(m.box.map50), "map": float(m.box.map),
            "precision": float(m.box.mp), "recall": float(m.box.mr)}


def false_alarms(weights, neg_list, imgsz, device, conf):
    """How many of the site's 'not fire' images the model still detects fire / smoke on."""
    paths = [p for p in Path(neg_list).read_text(encoding="utf-8").splitlines() if p and Path(p).exists()]
    if not paths:
        return {"false_alarms": 0, "neg_images": 0}
    from ultralytics import YOLO
    m = YOLO(str(weights))
    hits = 0
    for k in range(0, len(paths), 16):
        for r in m.predict(paths[k:k + 16], imgsz=imgsz, device=device, conf=conf, verbose=False):
            hits += int(r.boxes is not None and len(r.boxes) > 0)
    return {"false_alarms": hits, "neg_images": len(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detect", "classify"], default="detect")
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True, help="starting weights (current model or base model)")
    ap.add_argument("--current", default="", help="currently deployed model for comparison")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-list", default="", help="file listing 'not fire' images for the false-alarm check")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, device=a.device, project=a.project,
                name=a.name, exist_ok=True, workers=0, patience=max(5, a.epochs // 3), plots=False,
                verbose=True, cache=False, amp=True)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"

    print("[SafetyVision] evaluating new model...", flush=True)
    new = evaluate(a.task, best, a.data, a.imgsz, a.device)
    if a.neg_list:
        new.update(false_alarms(best, a.neg_list, a.imgsz, a.device, a.conf))
    old = None
    if a.current and Path(a.current).exists():
        print("[SafetyVision] evaluating current model on the same validation set...", flush=True)
        try:
            old = evaluate(a.task, a.current, a.data, a.imgsz, a.device)
            if a.neg_list:
                old.update(false_alarms(a.current, a.neg_list, a.imgsz, a.device, a.conf))
        except Exception as e:  # e.g. a user-supplied model with different classes
            print(f"[SafetyVision] could not evaluate current model: {e}", flush=True)
    metric = "accuracy" if a.task == "classify" else "mAP50"
    Path(a.out).write_text(json.dumps({"best": str(best), "new": new, "old": old, "metric": metric}, indent=2),
                           encoding="utf-8")
    print(f"[SafetyVision] done: new {metric}={new['score']:.3f}"
          f" current={old['score']:.3f}" if old else f"[SafetyVision] done: new {metric}={new['score']:.3f}", flush=True)


if __name__ == "__main__":
    main()
