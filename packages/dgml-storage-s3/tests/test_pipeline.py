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

"""A real DGML workspace with S3 blobs + local docs, driven through the public
API — proving S3BlobStore composes into the pipeline via ``ws.blobs``."""

from __future__ import annotations

import json
import shutil
from typing import Any

import pytest
from dgml_core import layout
from dgml_core.pages import GS_BINARIES
from dgml_core.storage import Workspace
from dgml_storage_s3 import S3BlobStore

needs_gs = pytest.mark.skipif(
    not any(shutil.which(b) for b in GS_BINARIES), reason="ghostscript not installed"
)


def _record_object_reads(ws: Workspace) -> list[tuple[str, str]]:
    """Record every object-read request the store actually puts on the wire.

    Hooked into boto3's event system rather than onto ``S3BlobStore``'s own
    methods, so the assertion is about real S3 traffic — including the
    ``HeadObject`` that ``download_file``'s transfer manager issues before
    fetching — and not about which helper we happened to call.

    Object keys are mapped back through the store's own ``_key`` so the recorded
    keys are workspace-relative: the fixtures give each test its own ``prefix``
    inside the shared session bucket, which is not part of what is asserted here.
    """
    seen: list[tuple[str, str]] = []
    store: Any = ws.blobs

    def hook(params: dict[str, Any], model: Any, **_: Any) -> None:
        if model.name in ("GetObject", "HeadObject"):
            seen.append((model.name, store._key(params.get("Key", ""))))

    store._s3.meta.events.register("before-parameter-build.s3.*", hook)
    return seen


def test_workspace_routes_blobs_to_s3_and_docs_to_local(s3_blobs_workspace: Workspace) -> None:
    from dgml_core.storage_local import LocalStore as _Local

    ws = s3_blobs_workspace
    assert isinstance(ws.blobs, S3BlobStore)
    assert isinstance(ws.docs, _Local)

    # A blob lands in S3 (invisible on the local root); a document lands locally.
    ws.blobs.put_blob("files/f1/page_images/page_1.png", b"\x89PNG")
    ws.docs.put_doc(layout.Collection.FILES, "f1", {"id": "f1"})
    assert not (ws.root / "files" / "f1" / "page_images" / "page_1.png").exists()
    assert (ws.root / "files" / "f1" / "file.json").is_file()
    assert ws.blobs.get_blob("files/f1/page_images/page_1.png") == b"\x89PNG"
    assert ws.docs.get_doc(layout.Collection.FILES, "f1") == {"id": "f1"}


@needs_gs
def test_file_add_renders_pages_into_s3(s3_blobs_workspace: Workspace) -> None:
    from dgml_core.files import FileStore

    ws = s3_blobs_workspace
    pdf = ws.root / "sample.pdf"
    _write_blank_pdf(pdf, pages=2)

    result = FileStore(ws).add(pdf)
    fid = result.record.id
    # Page images + page-text (blobs) were rendered into S3; the manifest (doc)
    # lives locally. This is the whole point: the pipeline never knew it wrote to
    # a remote object store.
    assert len(ws.blobs.list_blobs(layout.file_pages_prefix(fid))) == 2
    assert len(ws.blobs.list_blobs(layout.file_text_prefix(fid))) == 2
    assert ws.docs.get_doc(layout.Collection.FILES, fid) is not None
    assert not (ws.root / "files" / fid / "page_images").exists()  # not on local disk


def test_clustering_record_text_reads_only_page_text_from_s3(
    s3_blobs_workspace: Workspace,
) -> None:
    """A clustering record's ``text`` must not drag the file's whole S3 prefix down.

    ``WorkspaceFileDataset.__getitem__`` builds ``record.text`` from
    ``page_text/`` alone, but used to materialize all of ``files/<id>/`` to do it
    — the source document and every page render included, once per record per
    clustering pass. On ``LocalStore`` that is a zero-copy passthrough and costs
    nothing, which is exactly why it survived: the waste is invisible until the
    blobs are actually remote.

    Asserted against real S3 object reads, so the claim is about wire traffic.
    """
    from dgml_core.dataset import WorkspaceFileDataset

    ws = s3_blobs_workspace
    fid = "f1"
    ws.blobs.put_blob(layout.file_page_image_key(fid, 1), _png())
    # The stored original: the biggest object under the prefix and the clearest
    # thing that must never be fetched to assemble text.
    ws.blobs.put_blob(layout.file_source_key(fid, f"{fid}.pdf"), b"%PDF-1.7 " + b"x" * 50_000)
    for page in range(1, 4):
        ws.blobs.put_blob(
            layout.file_page_text_key(fid, page),
            json.dumps(
                {"page_number": page, "words": [{"t": f"w{page}", "l": [0, 0, 8, 12]}]}
            ).encode(),
        )

    seen = _record_object_reads(ws)
    record = WorkspaceFileDataset(ws, [fid], text_view="full")[0]

    assert record.text == "w1 w2 w3"  # every page still read
    keys = [key for _op, key in seen]
    assert keys, "no object reads recorded — the hook is not wired up"
    # The page render is fetched exactly once, by the image path (get_blob), and
    # never again as collateral of the text path.
    assert [k for k in keys if layout.PAGE_IMAGES_DIR in k] == [layout.file_page_image_key(fid, 1)]
    assert not [k for k in keys if k.endswith(".pdf")], "source document fetched to build text"


def _png() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(123, 200, 50)).save(buf, "PNG")
    return buf.getvalue()


def _write_blank_pdf(path: object, pages: int) -> None:
    # Minimal multi-page PDF (empty content stream per page).
    from pathlib import Path

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []

    def obj(body: bytes) -> int:
        offsets.append(len(out))
        num = len(offsets)
        out.extend(f"{num} 0 obj\n".encode() + body + b"\nendobj\n")
        return num

    obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    obj(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    for _ in range(pages):
        obj(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")
    xref = len(out)
    out.extend(f"xref\n0 {len(offsets) + 1}\n".encode() + b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    Path(str(path)).write_bytes(bytes(out))
