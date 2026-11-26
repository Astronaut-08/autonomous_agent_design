import time
from safety import LoopDetector, is_timed_out


def test_loop_detected_on_third_identical_call():
    d = LoopDetector(max_repeats=3)
    assert d.check("search_items", {"category": "desserts"}) is False
    assert d.check("search_items", {"category": "desserts"}) is False
    assert d.check("search_items", {"category": "desserts"}) is True


def test_different_args_no_loop():
    d = LoopDetector(max_repeats=3)
    for price in (100, 101, 102, 103):
        assert d.check("search_items", {"max_price": price}) is False


def test_key_order_and_nested_args_are_canonical():
    d = LoopDetector(max_repeats=2)
    d.check("calculate_order", {"items": [{"item_id": "DISH-010", "quantity": 1}], "x": 1})
    assert d.check("calculate_order", {"x": 1, "items": [{"quantity": 1, "item_id": "DISH-010"}]}) is True


def test_timeout():
    start = time.monotonic()
    assert is_timed_out(start, timeout=10) is False
    assert is_timed_out(start - 11, timeout=10) is True