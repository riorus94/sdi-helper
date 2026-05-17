"""Process-wide CLIP model cache shared by real-photo filter, view classifier, and embedding index."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("HF_HUB_TIMEOUT", "60")

_cache: dict[str, tuple[Any, Any]] = {}
# Text features are static (prompts never change at runtime) — cache them to avoid
# re-encoding the same prompt lists on every image call.
_text_cache: dict[tuple[str, ...], Any] = {}


def get_clip(model_name: str = "openai/clip-vit-base-patch32") -> tuple[Any, Any]:
    if model_name not in _cache:
        import torch  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor

        # Do NOT pass use_safetensors=True: on Windows, safetensors uses memory-mapped
        # files via safe_open which requires the pagefile to be large enough to cover the
        # full model size.  When multiple scrape processes run simultaneously (e.g. Task
        # Scheduler + manual) the pagefile exhausts and raises OSError 1455.  Letting
        # transformers auto-select the format avoids the mmap path on constrained systems.
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        _cache[model_name] = (model, processor)
    return _cache[model_name]


def clip_text_scores(img_bgr: Any, texts: list[str], model_name: str = "openai/clip-vit-base-patch32") -> Any:
    import numpy as np
    import torch
    from PIL import Image

    model, processor = get_clip(model_name)

    # Encode text once and cache — prompts are static across all images.
    cache_key = (model_name, *texts)
    if cache_key not in _text_cache:
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        _text_cache[cache_key] = text_features
    text_features = _text_cache[cache_key]

    # Encode image.
    pil = Image.fromarray(img_bgr[:, :, ::-1])
    image_inputs = processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity scaled by learned temperature, then softmax.
    logits = (image_features @ text_features.T) * model.logit_scale.exp()
    probs = logits.softmax(dim=1)
    return np.asarray(probs[0].detach().numpy())


def clip_image_embedding(img_bgr: Any, model_name: str = "openai/clip-vit-base-patch32") -> Any:
    import numpy as np
    import torch
    from PIL import Image

    model, processor = get_clip(model_name)
    pil = Image.fromarray(img_bgr[:, :, ::-1])
    inputs = processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return np.asarray(feats[0].cpu().numpy(), dtype=np.float32)
