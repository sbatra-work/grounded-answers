from grounded_answers.citations import split_claims, verify_citations

ALLOWED = {"leave-policy#1", "leave-policy#2"}


def test_valid_citations_are_kept() -> None:
    report = verify_citations("Staff get 25 days [leave-policy#1].", ALLOWED)
    assert report.text == "Staff get 25 days [leave-policy#1]."
    assert report.cited_ids == ["leave-policy#1"]
    assert report.dropped_claims == []


def test_citation_after_the_full_stop_stays_with_its_sentence() -> None:
    answer = "Staff get 25 days. [leave-policy#1] Sick leave is 10 days. [leave-policy#2]"
    assert split_claims(answer) == [
        "Staff get 25 days. [leave-policy#1]",
        "Sick leave is 10 days. [leave-policy#2]",
    ]


def test_claim_without_citation_is_stripped() -> None:
    answer = "Staff get 25 days [leave-policy#1]. The CEO earns a million euros."
    report = verify_citations(answer, ALLOWED)
    assert report.text == "Staff get 25 days [leave-policy#1]."
    assert report.dropped_claims == ["The CEO earns a million euros."]


def test_citation_outside_retrieved_set_is_stripped() -> None:
    answer = "Staff get 25 days [leave-policy#1]. Pets are welcome [pets#9]."
    report = verify_citations(answer, ALLOWED)
    assert report.cited_ids == ["leave-policy#1"]
    assert report.dropped_claims == ["Pets are welcome [pets#9]."]


def test_claim_mixing_real_and_invented_ids_is_stripped() -> None:
    report = verify_citations("Leave is generous [leave-policy#1][made-up#3].", ALLOWED)
    assert not report.has_content
    assert report.cited_ids == []


def test_all_claims_invalid_leaves_nothing() -> None:
    report = verify_citations("Trust me. It is fine.", ALLOWED)
    assert not report.has_content
    assert len(report.dropped_claims) == 2


def test_cited_ids_are_unique_and_ordered() -> None:
    answer = "A [leave-policy#2]. B [leave-policy#1]. C [leave-policy#2]."
    assert verify_citations(answer, ALLOWED).cited_ids == ["leave-policy#2", "leave-policy#1"]
