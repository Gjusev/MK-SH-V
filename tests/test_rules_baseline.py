"""Frozen-rules comparator tests: dev-frozen constants and rule behaviour."""

from __future__ import annotations

from mksh_v.rules_baseline import (
    RULES_DEV_ACCURACY,
    extract_total,
    find_amounts,
    parse_question_threshold,
    predict_rule,
    rule_doc_type,
    rule_field_threshold,
    rule_form_complete,
)


def test_amounts_parse_eu_and_us_formats():
    assert find_amounts("Subtotal 1.027,73 Tax") == [1027.73]
    assert find_amounts("Total 1,027.73 EUR") == [1027.73]
    assert find_amounts("OCR merge 2.656.28") == [2656.28]
    assert find_amounts("no amounts here") == []
    assert parse_question_threshold("Is the total at least 118.00 EUR?") == 118.0


def test_extract_total_prefers_hint_line_over_larger_amount():
    text = "Big distractor 999.99\nTOTAL  94.75\nTax 15.31"
    assert extract_total(text) == 94.75
    assert extract_total("only one line 12.34") == 12.34
    assert extract_total("nothing numeric") is None


def test_field_threshold_rule_decides_by_comparison():
    q = "Is the total on this document at least 100.00 EUR?"
    assert rule_field_threshold(q, "TOTAL 122.70") == "true"
    assert rule_field_threshold(q, "TOTAL 99.50") == "false"
    assert rule_field_threshold(q, "unreadable page") == "false"  # frozen fallback
    assert rule_field_threshold("no number", "TOTAL 50.00") == "false"


def test_doc_type_rule_order_is_specific_first():
    assert rule_doc_type("Maintenance Contract No. 68/2026\nThis agreement is entered into...") == "contract"
    assert rule_doc_type("INVOICE\nBilled to: Acme\nSubtotal 10.00") == "invoice"  # invoice beats subtotal
    assert rule_doc_type("NORTHWIND\nSubtotal 43.91\nTax 8.62") == "receipt"
    assert rule_doc_type("REQUEST FORM\nPostal code *") == "form"
    assert rule_doc_type("Dear colleague,\n...") == "letter"
    assert rule_doc_type("STORAGE NOTICE\nArea B closed") == "other"


def test_form_complete_rule_garbled_and_empty_star():
    # 'empty required' = starred line of <= 2 tokens (one-word label + star)
    assert rule_form_complete("REQUEST FORM\nSupervisor *\nFull name * Mira", 90.0) == "missing_required"
    assert rule_form_complete("REQUEST FORM\nPostal code * 41092\nFull name * Mira", 90.0) == "complete"
    assert rule_form_complete("REQUEST FORM\n�Supervisor *\nvalues detached", 90.0) == "illegible"
    assert rule_form_complete("REQUEST FORM\nFull name * Mira Lane\nLong enough filler text to pass the word bar easily here", 80.0) == "illegible"


def test_predict_rule_dispatch_and_none():
    assert predict_rule("field_threshold", "at least 10.00?", "TOTAL 12.00", 90.0) == "true"
    assert predict_rule("doc_type", "q", "INVOICE ...", 90.0) == "invoice"
    assert predict_rule("urgency", "q", "text", 90.0) is None  # no rule -> None


def test_dev_accuracies_are_recorded_above_majority():
    # frozen constants must stay the frozen constants
    assert RULES_DEV_ACCURACY["field_threshold"] == 0.960
    assert RULES_DEV_ACCURACY["doc_type"] == 0.767
    assert RULES_DEV_ACCURACY["form_complete"] == 0.440
    assert RULES_DEV_ACCURACY["form_complete"] > 1 / 3  # beats majority
