from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial", size=10, color="222222")


def _style_sheet(sheet, table_name: str) -> None:
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = BODY_FONT
            cell.alignment = Alignment(vertical="center")
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"
    for column in sheet.columns:
        letter = column[0].column_letter
        header = str(column[0].value or "")
        width = max((len(str(cell.value or "")) for cell in column), default=8) + 2
        sheet.column_dimensions[letter].width = min(60 if header == "job_ids" else 32, max(10, width))
    if sheet.max_row > 1 and sheet.max_column > 0:
        table = Table(displayName=table_name, ref=sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        sheet.add_table(table)


def write_table_workbook(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    sheet_name: str,
) -> Path:
    """Write one former CSV output as one formatted Excel table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    if not records:
        return path

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    headers = list(records[0].keys())
    sheet.append(headers)
    for record in records:
        sheet.append([record.get(header, "") for header in headers])
    _style_sheet(sheet, f"{sheet_name.replace(' ', '')}Table")
    workbook.save(path)

    # Remove the exact legacy CSV counterpart after its XLSX replacement exists.
    path.with_suffix(".csv").unlink(missing_ok=True)
    return path
