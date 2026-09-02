from __future__ import annotations

import base64
import io
import time
from abc import ABC, abstractmethod

import httpx
from PIL import Image, ImageOps

from .config import ARK_IMAGE_BASE_URL, ENV, FAL_EDIT_MODELS, PROVIDER_SETTINGS
from .image_ops import image_bytes, illustrated_patch


EDIT_PROMPT = (
    "Edit only the visible human head, hair, ears, face, and the small neck transition inside the target area. "
    "Replace the real identity with an original subtle anime-inspired illustrated face, not a sticker or emoji. "
    "Keep the same head direction, approximate hair silhouette, lighting, scale, and occlusion. "
    "Do not preserve biometric facial likeness. Do not edit clothing, body, furniture, architecture, text, or background. "
    "Use restrained linework, natural skin and hair colors, adult-like proportions and a seamless photo-to-illustration transition at the neck. "
    "Avoid oversized eyes, flat icon shapes, stickers, emoji, logos, celebrity likeness, and named artist styles."
)

# Two deliberately narrow profiles for the privacy-vs-naturalness experiment.
# Both retain scene/body pixels through local compositing; the difference is
# only how strongly the generated facial identity is abstracted.
BALANCED_PORTRAIT_PROMPT = (
    "Edit only the visible human head, hair, ears, face, and a small neck transition inside the target area. "
    "Create an original semi-realistic illustrated identity replacement that is clearly non-photorealistic and not recognizably the same person. "
    "Preserve head angle, hair silhouette and color, glasses or headwear, lighting, scale, and occlusion. "
    "Deliberately redesign facial geometry, eye shape and spacing, nose, mouth, jaw, and expression; do not preserve biometric likeness. "
    "Keep clothing, body, furniture, architecture, text, and background unchanged. "
    "Use restrained soft painterly linework, natural colors, adult proportions, and a seamless neck transition into the live-action body. "
    "Avoid photorealistic facial texture, oversized anime eyes, flat cartoons, icons, stickers, emoji, logos, celebrity likeness, and named artist styles."
)

BALANCED_PAINTERLY_PROMPT = (
    "Edit only the visible human head, hair, ears, face, and a small neck transition inside the target area. "
    "Create an original editorial-style illustrated identity replacement: one clear step more illustrated than a photograph, yet subtle and believable in this real scene. "
    "Preserve head angle, hair silhouette and color, glasses or headwear, lighting, scale, and occlusion. "
    "Replace all identifiable facial geometry with simplified painterly facial planes: deliberately change eye shape and spacing, nose, mouth, jaw, and expression so the person cannot be recognized. "
    "Keep clothing, body, furniture, architecture, text, and background unchanged. "
    "Use natural colors, adult proportions, light brush texture, and a soft seamless neck transition into the live-action body. "
    "Avoid photorealistic facial texture, oversized anime eyes, flat cartoons, icons, stickers, emoji, logos, celebrity likeness, and named artist styles."
)

# Provider adapters use this when a request does not intentionally select a
# stronger experimental profile.  Keep the legacy wording above only as a
# readable history of the first prototype prompt.
EDIT_PROMPT = BALANCED_PORTRAIT_PROMPT

PROMPT_PROFILES = {
    # The first 20-image run validated this profile as the user-approved
    # default: recognizably illustrated while retaining scene coherence.
    "default": EDIT_PROMPT,
    "balanced_portrait": BALANCED_PORTRAIT_PROMPT,
    "balanced_painterly": BALANCED_PAINTERLY_PROMPT,
}


def prompt_for_profile(profile: str) -> str:
    return PROMPT_PROFILES.get(profile, BALANCED_PORTRAIT_PROMPT)


class ProviderError(RuntimeError):
    pass


class ImageProvider(ABC):
    name: str

    @abstractmethod
    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        raise NotImplementedError


class LocalIllustrationProvider(ImageProvider):
    name = "local"

    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        # The service passes crop-relative head bounds through mask.getbbox().
        bbox = mask.getbbox() or (0, 0, crop.width, crop.height)
        x1, y1, x2, y2 = bbox
        return illustrated_patch(crop, [0, 0, crop.width, crop.height], [x1, y1, x2 - x1, y2 - y1], subject_id, retry_nonce)


class OpenAIProvider(ImageProvider):
    name = "openai"

    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        settings = PROVIDER_SETTINGS[self.name]
        key = ENV.get(settings.key_name or "")
        if not key:
            raise ProviderError("OPENAI_API_KEY 未配置。")
        crop_data = image_bytes(crop, "PNG")
        alpha = ImageOps.invert(mask.convert("L"))
        mask_rgba = Image.new("RGBA", crop.size, (255, 255, 255, 255))
        mask_rgba.putalpha(alpha)
        mask_buffer = io.BytesIO()
        mask_rgba.save(mask_buffer, format="PNG")
        files = {
            "image[]": ("crop.png", crop_data, "image/png"),
            "mask": ("mask.png", mask_buffer.getvalue(), "image/png"),
        }
        data = {"model": settings.model, "prompt": prompt, "quality": "high", "input_fidelity": "high"}
        response = httpx.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {key}"}, files=files, data=data, timeout=180,
        )
        _raise_for_provider(response, "OpenAI")
        payload = response.json()
        encoded = payload.get("data", [{}])[0].get("b64_json")
        if not encoded:
            raise ProviderError("OpenAI 未返回图像数据。")
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


class GeminiProvider(ImageProvider):
    name = "gemini"

    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        settings = PROVIDER_SETTINGS[self.name]
        key = ENV.get(settings.key_name or "")
        if not key:
            raise ProviderError("GEMINI_API_KEY 未配置。")
        encoded = base64.b64encode(image_bytes(crop, "PNG")).decode("ascii")
        payload = {
            "model": settings.model,
            "input": [
                {"type": "image", "mime_type": "image/png", "data": encoded},
                {"type": "text", "text": prompt},
            ],
            "response_format": {"type": "image", "image_size": "1K"},
        }
        response = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload, timeout=180,
        )
        _raise_for_provider(response, "Gemini")
        data = response.json()
        image_data = _find_base64_image(data)
        if not image_data:
            raise ProviderError("Gemini 未返回图像数据。")
        return Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")


class FalProvider(ImageProvider):
    """fal.ai server-side adapter for local crop + mask image editing."""

    name = "fal"

    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        settings = PROVIDER_SETTINGS[self.name]
        key = ENV.get(settings.key_name or "")
        if not key:
            raise ProviderError("FAL_KEY 未配置。")

        endpoint = settings.model.strip().strip("/")
        profile = FAL_EDIT_MODELS.get(endpoint)
        if not profile:
            raise ProviderError("该 fal.ai 端点尚未经过本地隐私合成兼容性验证。")

        crop_uri = _data_uri(image_bytes(crop, "PNG"), "image/png")
        mask_uri = _data_uri(image_bytes(mask.convert("L"), "PNG"), "image/png")
        payload: dict[str, object] = {
            "prompt": prompt,
            "image_urls": [crop_uri],
            "num_images": 1,
            "output_format": "png",
            "sync_mode": True,
        }
        if profile["adapter"] == "nano_banana":
            payload.update({"aspect_ratio": "auto", "safety_tolerance": "4"})
        elif profile["adapter"] == "gpt_image":
            payload.update({
                "image_size": "auto",
                "quality": "high",
                "mask_image_url": mask_uri,
            })
        elif profile["adapter"] == "seedream":
            # Seedream's fal edit endpoints use natural-language image editing
            # without a mask parameter. The crop plus final local compositing
            # remains the mandatory privacy boundary.
            payload.pop("output_format")
            payload.update({
                "image_size": "auto_2K",
                "max_images": 1,
                "enable_safety_checker": True,
            })

        response = httpx.post(
            f"https://fal.run/{endpoint}",
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=240,
        )
        _raise_for_provider(response, "fal.ai")
        try:
            url = response.json()["images"][0]["url"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("fal.ai 未返回图像数据。") from exc
        return _load_provider_image(url)


class ArkSeedreamProvider(ImageProvider):
    """Volcengine Ark Agent Plan Seedream adapter.

    Agent Plan has a dedicated /api/plan/v3 endpoint.  It is intentionally
    separate from both fal and Ark's normal /api/v3 endpoint, which would use
    different billing.
    """

    name = "ark"

    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        settings = PROVIDER_SETTINGS[self.name]
        key = ENV.get(settings.key_name or "")
        if not key:
            raise ProviderError("ARK_API_KEY 未配置。")
        crop_uri = _data_uri(image_bytes(crop, "PNG"), "image/png")
        payload = {
            "model": settings.model,
            "prompt": prompt,
            "image": crop_uri,
            "size": "2K",
            "response_format": "url",
        }
        response = httpx.post(
            ARK_IMAGE_BASE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=240,
        )
        _raise_for_provider(response, "火山方舟")
        payload = response.json()
        try:
            item = payload["data"][0]
            url = item.get("url")
            encoded = item.get("b64_json")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("火山方舟未返回图像数据。") from exc
        if encoded:
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        if not url:
            raise ProviderError("火山方舟未返回图像 URL。")
        return _load_provider_image(url)


class BFLProvider(ImageProvider):
    name = "bfl"

    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        settings = PROVIDER_SETTINGS[self.name]
        key = ENV.get(settings.key_name or "")
        if not key:
            raise ProviderError("BFL_API_KEY 未配置。")
        encoded = base64.b64encode(image_bytes(crop, "JPEG", quality=95)).decode("ascii")
        model_endpoint = settings.model if settings.model.startswith("flux-") else "flux-2-pro"
        response = httpx.post(
            f"https://api.bfl.ai/v1/{model_endpoint}",
            headers={"x-key": key, "Content-Type": "application/json", "accept": "application/json"},
            json={"prompt": prompt, "input_image": f"data:image/jpeg;base64,{encoded}"}, timeout=60,
        )
        _raise_for_provider(response, "BFL")
        payload = response.json()
        polling_url = payload.get("polling_url")
        if not polling_url:
            raise ProviderError("BFL 未返回轮询地址。")
        for _ in range(180):
            poll = httpx.get(polling_url, headers={"x-key": key, "accept": "application/json"}, timeout=30)
            _raise_for_provider(poll, "BFL")
            status = poll.json()
            if status.get("status") == "Ready":
                url = status.get("result", {}).get("sample")
                if not url:
                    break
                image_response = httpx.get(url, timeout=60, follow_redirects=True)
                image_response.raise_for_status()
                return Image.open(io.BytesIO(image_response.content)).convert("RGB")
            if status.get("status") in {"Error", "Failed"}:
                raise ProviderError(f"BFL 生成失败：{status}")
            time.sleep(1)
        raise ProviderError("BFL 生成超时。")


class QwenProvider(ImageProvider):
    name = "qwen"

    def edit(self, crop: Image.Image, mask: Image.Image, *, subject_id: str, retry_nonce: int, prompt: str = EDIT_PROMPT) -> Image.Image:
        settings = PROVIDER_SETTINGS[self.name]
        key = ENV.get(settings.key_name or "")
        if not key:
            raise ProviderError("DASHSCOPE_API_KEY 未配置。")
        encoded = base64.b64encode(image_bytes(crop, "JPEG", quality=95)).decode("ascii")
        payload = {
            "model": settings.model,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{encoded}"},
                        {"text": prompt},
                    ],
                }]
            },
            "parameters": {"n": 1, "prompt_extend": False, "watermark": False},
        }
        response = httpx.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload, timeout=180,
        )
        _raise_for_provider(response, "Qwen")
        data = response.json()
        try:
            url = data["output"]["choices"][0]["message"]["content"][0]["image"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Qwen 未返回图像地址。") from exc
        image_response = httpx.get(url, timeout=60, follow_redirects=True)
        image_response.raise_for_status()
        return Image.open(io.BytesIO(image_response.content)).convert("RGB")


def _raise_for_provider(response: httpx.Response, label: str) -> None:
    if response.is_success:
        return
    detail = response.text[:600]
    raise ProviderError(f"{label} API 返回 {response.status_code}：{detail}")


def _find_base64_image(value) -> str | None:
    if isinstance(value, dict):
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            return value["data"]
        if isinstance(value.get("output_image"), dict) and isinstance(value["output_image"].get("data"), str):
            return value["output_image"]["data"]
        for child in value.values():
            found = _find_base64_image(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_base64_image(child)
            if found:
                return found
    return None


def _data_uri(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _load_provider_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        try:
            encoded = url.split(",", 1)[1]
        except IndexError as exc:
            raise ProviderError("图像 data URI 格式无效。") from exc
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    image_response = httpx.get(url, timeout=60, follow_redirects=True)
    image_response.raise_for_status()
    return Image.open(io.BytesIO(image_response.content)).convert("RGB")


PROVIDERS: dict[str, ImageProvider] = {
    "local": LocalIllustrationProvider(),
    "openai": OpenAIProvider(),
    "gemini": GeminiProvider(),
    "fal": FalProvider(),
    "ark": ArkSeedreamProvider(),
    "bfl": BFLProvider(),
    "qwen": QwenProvider(),
}
