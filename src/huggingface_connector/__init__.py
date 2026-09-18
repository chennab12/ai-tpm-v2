"""HuggingFace model connector."""

from dataclasses import dataclass

import httpx


@dataclass
class HFModelInfo:
    """Metadata about a HuggingFace model."""
    model_id: str
    model_type: str
    pipeline_tag: str
    library_name: str
    downloads: int
    likes: int
    tags: list[str]


class HuggingFaceConnector:
    """Query HuggingFace Hub for model metadata."""

    BASE_URL = "https://huggingface.co/api/models"

    def __init__(self, token: str = ""):
        self.token = token
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def get_model_info(self, model_id: str) -> HFModelInfo | None:
        """Fetch model metadata from HF Hub."""
        try:
            resp = httpx.get(
                f"{self.BASE_URL}/{model_id}",
                headers=self.headers,
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return HFModelInfo(
                model_id=data.get("id", model_id),
                model_type=data.get("pipeline_tag", "unknown"),
                pipeline_tag=data.get("pipeline_tag", "text-generation"),
                library_name=data.get("library_name", "unknown"),
                downloads=data.get("downloads", 0),
                likes=data.get("likes", 0),
                tags=data.get("tags", []),
            )
        except Exception:
            return None

    def search_models(
        self, query: str, limit: int = 10, sort: str = "downloads"
    ) -> list[dict]:
        """Search for models on HF Hub."""
        try:
            resp = httpx.get(
                self.BASE_URL,
                params={"search": query, "limit": limit, "sort": sort, "direction": -1},
                headers=self.headers,
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            return [
                {
                    "model_id": m.get("id"),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                }
                for m in resp.json()
            ]
        except Exception:
            return []

    def get_model_size_estimate(self, model_id: str) -> dict:
        """Estimate model size from config."""
        info = self.get_model_info(model_id)
        if not info:
            return {"model_id": model_id, "estimated_params": "unknown", "gpu_needed": "unknown"}

        tags = info.tags
        params = "unknown"
        gpu = "unknown"

        for tag in tags:
            if "7b" in tag.lower():
                params, gpu = "7B", "16GB+ VRAM"
            elif "13b" in tag.lower():
                params, gpu = "13B", "32GB+ VRAM"
            elif "70b" in tag.lower():
                params, gpu = "70B", "80GB+ VRAM"
            elif "8b" in tag.lower():
                params, gpu = "8B", "16GB+ VRAM"

        return {
            "model_id": model_id,
            "estimated_params": params,
            "gpu_needed": gpu,
            "downloads": info.downloads,
            "library": info.library_name,
        }
