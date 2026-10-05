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

"""Tests for the provided converters (command, xlsx, libreoffice, aspose)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dgml_core.conversion import ConverterConfig
from dgml_core.errors import ConversionConfigInvalid, ConversionFailed
from translators_pdf.aspose import AsposeCellsConverter, AsposeWordsConverter
from translators_pdf.command import CommandConverter
from translators_pdf.libreoffice import LibreOfficeConverter, _find_soffice
from translators_pdf.xlsx import XlsxIslandsConverter

_CMD = "translators_pdf.command:CommandConverter"


def _cmd_config(command: list[str], **extra: object) -> ConverterConfig:
    return CommandConverter.parse_config({"provider": _CMD, "command": command, **extra})


# --- command converter --------------------------------------------------------


def test_command_requires_input_token() -> None:
    with pytest.raises(ConversionConfigInvalid):
        CommandConverter.parse_config({"provider": _CMD, "command": ["x", "{output}"]})


def test_command_requires_exactly_one_output_token() -> None:
    with pytest.raises(ConversionConfigInvalid):
        CommandConverter.parse_config({"provider": _CMD, "command": ["x", "{input}"]})
    with pytest.raises(ConversionConfigInvalid):
        CommandConverter.parse_config(
            {"provider": _CMD, "command": ["x", "{input}", "{output}", "{output_dir}"]}
        )


def test_command_rejects_non_list_command() -> None:
    with pytest.raises(ConversionConfigInvalid):
        CommandConverter.parse_config({"provider": _CMD, "command": "soffice {input}"})


def test_command_to_pdf_output_token(tmp_path: Path) -> None:
    src = tmp_path / "in.docx"
    src.write_bytes(b"src")
    writer = "import sys,pathlib; pathlib.Path(sys.argv[1]).write_bytes(b'%PDF-1.4 out')"
    conv = CommandConverter(_cmd_config([sys.executable, "-c", writer, "{output}", "{input}"]))
    assert conv.to_pdf(src) == b"%PDF-1.4 out"


def test_command_to_pdf_output_dir_token(tmp_path: Path) -> None:
    src = tmp_path / "in.xlsx"
    src.write_bytes(b"src")
    writer = (
        "import sys,pathlib; pathlib.Path(sys.argv[1], 'result.pdf').write_bytes(b'%PDF-1.4 dir')"
    )
    conv = CommandConverter(_cmd_config([sys.executable, "-c", writer, "{output_dir}", "{input}"]))
    assert conv.to_pdf(src) == b"%PDF-1.4 dir"


def test_command_missing_binary_raises(tmp_path: Path) -> None:
    src = tmp_path / "in.docx"
    src.write_bytes(b"src")
    conv = CommandConverter(_cmd_config(["dgml-no-such-binary-xyz", "{input}", "{output}"]))
    with pytest.raises(ConversionFailed):
        conv.to_pdf(src)


def test_command_no_output_produced_raises(tmp_path: Path) -> None:
    src = tmp_path / "in.docx"
    src.write_bytes(b"src")
    # A command that exits cleanly but writes nothing.
    conv = CommandConverter(_cmd_config([sys.executable, "-c", "pass", "{input}", "{output}"]))
    with pytest.raises(ConversionFailed):
        conv.to_pdf(src)


# --- xlsx converter -----------------------------------------------------------


def test_xlsx_orientation_validation() -> None:
    with pytest.raises(ConversionConfigInvalid):
        XlsxIslandsConverter.parse_config({"provider": "p", "orientation": "sideways"})


def test_xlsx_gap_validation() -> None:
    with pytest.raises(ConversionConfigInvalid):
        XlsxIslandsConverter.parse_config({"provider": "p", "row_gap": -1})


def test_xlsx_to_pdf_renders(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("reportlab")
    src = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet["A1"] = "Name"
    sheet["B1"] = "Amount"
    sheet["A2"] = "Widget"
    sheet["B2"] = 42
    wb.save(src)

    conv = XlsxIslandsConverter(
        XlsxIslandsConverter.parse_config({"provider": "p", "orientation": "portrait"})
    )
    data = conv.to_pdf(src)
    assert data.startswith(b"%PDF")


def test_xlsx_to_pdf_splits_many_rows(tmp_path: Path) -> None:
    """A tall island (many rows) splits across pages via LongTable instead of
    failing to fit one frame."""
    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("reportlab")
    src = tmp_path / "tall.xlsx"
    wb = openpyxl.Workbook()
    sheet = wb.active
    for r in range(1, 201):
        sheet.cell(r, 1, f"row {r}")
        sheet.cell(r, 2, r)
    wb.save(src)

    conv = XlsxIslandsConverter(XlsxIslandsConverter.parse_config({"provider": "p"}))
    assert conv.to_pdf(src).startswith(b"%PDF")


def test_xlsx_oversized_island_raises_actionable_error(tmp_path: Path) -> None:
    """A single row too tall to fit a page yields a clear ConversionFailed,
    not reportlab's opaque 'NoneType > float'."""
    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("reportlab")
    src = tmp_path / "huge-cell.xlsx"
    wb = openpyxl.Workbook()
    # One cell with enough text that, wrapped in a normal column, the single
    # row is taller than a page — unsplittable.
    wb.active["A1"] = "lorem ipsum dolor " * 4000
    wb.save(src)

    conv = XlsxIslandsConverter(XlsxIslandsConverter.parse_config({"provider": "p"}))
    with pytest.raises(ConversionFailed) as excinfo:
        conv.to_pdf(src)
    message = str(excinfo.value)
    assert "too large or too wide to fit on a page" in message
    # The clean message (before the underlying-error suffix) must not leak the
    # opaque reportlab TypeError.
    assert "NoneType" not in message.split("Underlying error:")[0]


# --- libreoffice / aspose: lazy-failure when the tool is absent ----------------


def test_libreoffice_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("translators_pdf.libreoffice._find_soffice", lambda: None)
    with pytest.raises(ConversionFailed):
        LibreOfficeConverter(ConverterConfig(provider="p"))


def test_aspose_not_installed_raises() -> None:
    if "aspose.words" in sys.modules:  # pragma: no cover - only if someone installs it
        pytest.skip("aspose-words is installed in this environment")
    with pytest.raises(ConversionFailed):
        AsposeWordsConverter(ConverterConfig(provider="p"))


def test_aspose_cells_not_installed_raises() -> None:
    if "aspose.cells" in sys.modules:  # pragma: no cover - only if someone installs it
        pytest.skip("aspose-cells-python is installed in this environment")
    with pytest.raises(ConversionFailed):
        AsposeCellsConverter(ConverterConfig(provider="p"))


def test_aspose_cells_license_must_be_non_empty_string() -> None:
    with pytest.raises(ConversionConfigInvalid):
        AsposeCellsConverter.parse_config({"provider": "p", "license": ""})


def test_find_soffice_returns_str_or_none() -> None:
    found = _find_soffice()
    assert found is None or isinstance(found, str)


def test_xlsx_islands_ignore_a_stray_cell_far_from_the_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One formatted, empty cell at HV65465 stretches the sheet's declared
    range to 65,465 rows by 230 columns. Island detection must read the
    handful of cells the sheet has, not allocate the whole rectangle: any
    call to ``sheet.cell`` is the walk starting, and fails the test, and the
    cell map must be the same size afterwards."""
    openpyxl = pytest.importorskip("openpyxl")
    from translators_pdf import _xlsx_detector

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet["A1"] = "Invoice"
    sheet["A2"] = "Widget"
    sheet["B2"] = 42
    sheet["HV65465"].number_format = "0.00"  # exists, empty, far away
    assert sheet.max_row == 65465 and sheet.max_column == 230
    cells_before = len(sheet._cells)

    def _walk_started(*args: object, **kwargs: object) -> None:
        pytest.fail("find_islands walked the declared range through sheet.cell()")

    monkeypatch.setattr(sheet, "cell", _walk_started)
    assert _xlsx_detector.find_islands(sheet) == [(1, 1, 2, 2)]
    assert len(sheet._cells) == cells_before


def test_xlsx_islands_fall_back_to_the_walk_for_a_foreign_sheet() -> None:
    """An object that is not an openpyxl worksheet is walked as before, even
    when it carries an attribute called ``_cells``; hidden rows and columns
    are skipped before any cell is read, as before."""
    from types import SimpleNamespace

    from translators_pdf import _xlsx_detector

    walked: list[tuple[int, int]] = []

    def cell(row: int, column: int) -> SimpleNamespace:
        walked.append((row, column))
        return SimpleNamespace(value="x" if (row, column) != (2, 2) else None)

    sheet = SimpleNamespace(
        max_row=3,
        max_column=2,
        row_dimensions={3: SimpleNamespace(hidden=True)},
        column_dimensions={},
        _cells={"metadata": object()},
        cell=cell,
    )
    assert _xlsx_detector.find_islands(sheet) == [(1, 1, 2, 2)]
    assert walked == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_xlsx_islands_still_skip_hidden_rows_and_columns() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from translators_pdf import _xlsx_detector

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet["A1"] = "keep"
    sheet["A2"] = "hidden row"
    sheet["C1"] = "hidden col"
    sheet.row_dimensions[2].hidden = True
    sheet.column_dimensions["C"].hidden = True
    assert _xlsx_detector.find_islands(sheet) == [(1, 1, 1, 1)]
