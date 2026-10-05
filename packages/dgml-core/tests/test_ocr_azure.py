# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AzureProvider tests with mocked DocumentIntelligenceClient."""

from __future__ import annotations

import json
import sys
import threading
import types
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from dgml_core.errors import AuthError, MissingExtra, OcrFailed
from dgml_core.ocr import OcrConfig, OcrProviderName, extract_text_ocr
from dgml_core.storage import Workspace

from .conftest import make_fake_png


class _FakePoller:
    def __init__(self, result: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class _FakeAzureClient:
    """Stand-in for DocumentIntelligenceClient that returns a result
    keyed by a unique substring in the input bytes, so the test is
    order-independent under parallel dispatch."""

    def __init__(self, results_by_marker: dict[bytes, Any]) -> None:
        self._results = results_by_marker
        self._lock = threading.Lock()
        self.call_count = 0

    def begin_analyze_document(self, model_id: str, *, body: Any) -> _FakePoller:
        bytes_in = body.read() if hasattr(body, "read") else body
        with self._lock:
            self.call_count += 1
        for marker, result in self._results.items():
            if marker in bytes_in:
                return _FakePoller(result)
        raise KeyError(f"no fake result for body {bytes_in[:32]!r}")


def _azure_word(text: str, polygon: list[float]) -> types.SimpleNamespace:
    return types.SimpleNamespace(content=text, polygon=polygon)


def _azure_page_pixel(
    width_px: int, height_px: int, words: list[Any], angle: float | None = None
) -> Any:
    """Build a fake page like the SDK returns for image input (unit='pixel')."""
    return types.SimpleNamespace(
        page_number=1,
        width=float(width_px),
        height=float(height_px),
        unit="pixel",
        angle=angle,
        words=words,
    )


class _FixedAzureClient:
    """Fake client that returns one fixed result regardless of input bytes —
    for single-page tests where marker-routing (which needs the marker to be a
    substring of the input) would fight a real, PIL-decodable PNG."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.call_count = 0

    def begin_analyze_document(self, model_id: str, *, body: Any) -> _FakePoller:
        if hasattr(body, "read"):
            body.read()
        self.call_count += 1
        return _FakePoller(self._result)


def _azure_result(page: Any) -> Any:
    return types.SimpleNamespace(pages=[page])


def test_azure_missing_env_var_raises_auth_error(
    workspace: Workspace, text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_AZURE_KEY", raising=False)
    # Provide a non-empty page_images dir so we don't trip the
    # "no page images" guard before hitting the auth path.
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(100, 100))
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    with pytest.raises(AuthError, match="TEST_AZURE_KEY"):
        extract_text_ocr(
            text_pdf,
            workspace.root / "page_text",
            file_id="does-not-matter",
            page_images_dir=pages_dir,
            config=cfg,
        )


def test_azure_literal_api_key_builds_key_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A literal `api_key` in config should construct an AzureKeyCredential
    directly, without consulting os.environ."""
    from dgml_core.ocr_azure import _azure_credential

    # Make doubly sure the env-var path isn't accidentally taken.
    monkeypatch.delenv("ANY_KEY_ENV", raising=False)
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key="literal-key-value",
    )
    cred = _azure_credential(cfg)
    # AzureKeyCredential exposes .key.
    assert getattr(cred, "key", None) == "literal-key-value"


def test_azure_missing_sdk_raises_missing_extra(
    workspace: Workspace, text_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uninstalled `azure` extra raises :class:`MissingExtra`, naming the extra
    as a field rather than only in prose.

    It used to raise ``OcrFailed`` — the class for a provider/API failure — which
    conflated "your credentials or the service are bad" with "this machine never
    had the SDK". The message is unchanged; what is new is that a caller can read
    ``exc.extra`` and offer the install instead of regexing the text."""
    # Setting sys.modules[name] = None makes importlib treat the name as unimportable.
    monkeypatch.setitem(sys.modules, "azure.ai.documentintelligence", None)
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    with pytest.raises(MissingExtra, match="pip install dgml\\[azure\\]") as caught:
        extract_text_ocr(
            text_pdf,
            workspace.root / "page_text",
            file_id="does-not-matter",
            # page_images_dir is irrelevant — SDK import fails first.
            page_images_dir=workspace.root / "page_images",
            config=cfg,
        )
    assert caught.value.extra == "azure"
    assert caught.value.distribution == "azure-ai-documentintelligence"
    assert caught.value.code == "MISSING_EXTRA"


def test_azure_extract_writes_per_page_json(
    azure_config: Workspace,
    text_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")

    # Pretend ghostscript rendered two 1700x2200 page images. Real PNG
    # headers so _image_dimensions can parse them; the embedded marker
    # routes the fake Azure response.
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(1700, 2200, b"page-1-marker"))
    (pages_dir / "page_2.png").write_bytes(make_fake_png(1700, 2200, b"page-2-marker"))

    # Per-image dispatch → one response per call, keyed by a marker in
    # the input bytes. Page 1 has a word, page 2 is empty.
    results: dict[bytes, Any] = {
        b"page-1-marker": _azure_result(
            _azure_page_pixel(
                width_px=1700,
                height_px=2200,
                words=[
                    _azure_word("hello", [200.0, 300.0, 400.0, 300.0, 400.0, 360.0, 200.0, 360.0])
                ],
            )
        ),
        b"page-2-marker": _azure_result(_azure_page_pixel(width_px=1700, height_px=2200, words=[])),
    }

    captured: dict[str, Any] = {}
    client = _FakeAzureClient(results)

    def fake_ctor(endpoint: str, credential: Any, **kwargs: Any) -> _FakeAzureClient:
        captured["endpoint"] = endpoint
        captured["credential_type"] = type(credential).__name__
        captured["kwargs"] = kwargs
        return client

    import azure.ai.documentintelligence as adi

    monkeypatch.setattr(adi, "DocumentIntelligenceClient", fake_ctor)

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    result = extract_text_ocr(
        text_pdf,
        out_dir,
        file_id="fid",
        page_images_dir=pages_dir,
        config=cfg,
    )

    assert result.pages_written == 2
    assert result.pages_with_words == 1
    assert result.total_words == 1
    assert client.call_count == 2  # one Azure call per page image
    assert captured["endpoint"] == "https://example.cognitiveservices.azure.com/"
    assert "KeyCredential" in captured["credential_type"]

    # Page dims come from the PNG IHDR read by the shared loop (1700x2200
    # for our fake PNGs above). Word polygon passes through unchanged
    # since image input → unit='pixel'.
    p1 = json.loads((out_dir / "page_1.json").read_text())
    assert p1 == {
        "file_id": "fid",
        "page": 1,
        "width": 1700,
        "height": 2200,
        "words": [{"t": "hello", "l": [200, 300, 400, 360]}],
    }
    p2 = json.loads((out_dir / "page_2.json").read_text())
    assert p2["page"] == 2
    assert p2["words"] == []


def test_azure_rejects_unexpected_unit(
    azure_config: Workspace,
    text_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If Azure ever returns a non-pixel unit for image input, we'd
    rather fail loudly than silently scale-fudge."""
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")

    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(100, 100, b"page-1-marker"))

    weird_page = types.SimpleNamespace(
        page_number=1,
        width=1.0,
        height=1.0,
        unit="inch",  # unexpected for image input
        words=[],
    )
    client = _FakeAzureClient({b"page-1-marker": _azure_result(weird_page)})

    def fake_ctor(endpoint: str, credential: Any, **kwargs: Any) -> _FakeAzureClient:
        return client

    import azure.ai.documentintelligence as adi

    monkeypatch.setattr(adi, "DocumentIntelligenceClient", fake_ctor)

    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    with pytest.raises(OcrFailed, match="unexpected unit"):
        extract_text_ocr(
            text_pdf,
            tmp_path / "page_text",
            file_id="fid",
            page_images_dir=pages_dir,
            config=cfg,
        )


def test_azure_client_built_with_timeouts_and_retry(
    azure_config: Workspace,
    text_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The Azure client must be constructed with connection/read timeouts and
    retry — otherwise a wedged connection hangs a whole batch indefinitely."""
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(100, 100, b"page-1-marker"))

    results: dict[bytes, Any] = {
        b"page-1-marker": _azure_result(_azure_page_pixel(100, 100, words=[])),
    }
    client = _FakeAzureClient(results)
    captured: dict[str, Any] = {}

    def fake_ctor(endpoint: str, credential: Any, **kwargs: Any) -> _FakeAzureClient:
        captured.update(kwargs)
        return client

    import azure.ai.documentintelligence as adi

    monkeypatch.setattr(adi, "DocumentIntelligenceClient", fake_ctor)

    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    extract_text_ocr(
        text_pdf, tmp_path / "page_text", file_id="fid", page_images_dir=pages_dir, config=cfg
    )

    assert captured.get("connection_timeout") and captured["connection_timeout"] > 0
    assert captured.get("read_timeout") and captured["read_timeout"] > 0
    assert captured.get("retry_total", 0) >= 1


def test_azure_extract_requires_page_images(
    azure_config: Workspace,
    text_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    with pytest.raises(OcrFailed, match="no page images"):
        extract_text_ocr(
            text_pdf,
            tmp_path / "page_text",
            file_id="fid",
            page_images_dir=tmp_path / "page_images",  # does not exist
            config=cfg,
        )


def _real_png(width: int, height: int) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_azure_significant_angle_deskews_page(
    azure_config: Workspace,
    text_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When Azure reports a significant page angle, the page image is rewritten
    deskewed, the JSON records the rotation, and dims match the new image."""
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    original_png = _real_png(400, 200)
    (pages_dir / "page_1.png").write_bytes(original_png)

    word = _azure_word("skewed", [200.0, 50.0, 300.0, 50.0, 300.0, 90.0, 200.0, 90.0])
    result = _azure_result(_azure_page_pixel(400, 200, words=[word], angle=11.0))
    client = _FixedAzureClient(result)

    def fake_ctor(endpoint: str, credential: Any, **kwargs: Any) -> _FixedAzureClient:
        return client

    import azure.ai.documentintelligence as adi

    monkeypatch.setattr(adi, "DocumentIntelligenceClient", fake_ctor)

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    extract_text_ocr(text_pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg)

    # The canonical page image was rewritten (deskewed) in place.
    from PIL import Image

    rewritten = (pages_dir / "page_1.png").read_bytes()
    assert rewritten != original_png
    with Image.open(BytesIO(rewritten)) as im:
        img_w, img_h = im.width, im.height
    assert (img_w, img_h) != (400, 200)  # expand for |angle| >= 5

    payload = json.loads((out_dir / "page_1.json").read_text())
    assert payload["rotation"] == 11.0
    assert (payload["width"], payload["height"]) == (img_w, img_h)  # dims match image
    assert payload["words"], "the word should survive deskew"
    box = payload["words"][0]["l"]
    assert 0 <= box[0] < box[2] <= img_w
    assert 0 <= box[1] < box[3] <= img_h


def test_azure_small_angle_leaves_page_untouched(
    azure_config: Workspace,
    text_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A sub-threshold angle (<1°) is treated as upright: image untouched, no
    rotation key, dims unchanged."""
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    original_png = _real_png(400, 200)
    (pages_dir / "page_1.png").write_bytes(original_png)

    word = _azure_word("flat", [10.0, 20.0, 60.0, 20.0, 60.0, 40.0, 10.0, 40.0])
    result = _azure_result(_azure_page_pixel(400, 200, words=[word], angle=0.4))
    client = _FixedAzureClient(result)

    def fake_ctor(endpoint: str, credential: Any, **kwargs: Any) -> _FixedAzureClient:
        return client

    import azure.ai.documentintelligence as adi

    monkeypatch.setattr(adi, "DocumentIntelligenceClient", fake_ctor)

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    extract_text_ocr(text_pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg)

    assert (pages_dir / "page_1.png").read_bytes() == original_png  # untouched
    payload = json.loads((out_dir / "page_1.json").read_text())
    assert "rotation" not in payload
    assert (payload["width"], payload["height"]) == (400, 200)
    assert payload["words"][0]["l"] == [10, 20, 60, 40]  # box unchanged
