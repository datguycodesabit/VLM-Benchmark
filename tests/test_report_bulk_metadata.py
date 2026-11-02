import json

from openpyxl import load_workbook

from vlm_bench.reporting import render_report


def test_tokenizer_array_larger_than_excel_row_limit_is_preserved_compactly(tmp_path):
    values = [0] * 1_048_576
    report = {
        "conditions": {},
        "measurement_context": [{"tokenizer": {"token_type": values}}],
    }
    render_report(report, tmp_path, formats=("html", "xlsx", "json"))
    assert (tmp_path / "report.html").stat().st_size < 20000
    assert json.loads((tmp_path / "evaluation.json").read_text()) == report
    workbook = load_workbook(tmp_path / "report.xlsx", read_only=True)
    evidence = workbook["Evaluation model"]
    assert evidence.max_row < 10
    path, marker = next(evidence.iter_rows(min_row=2, values_only=True))
    assert path == "measurement_context[0].tokenizer.token_type"
    assert "FULL TEXT IN Text Overflow" in marker
    chunks = [row[-1] for row in workbook["Text Overflow"].iter_rows(min_row=2, values_only=True)]
    assert json.loads("".join(chunks)) == values
    workbook.close()
