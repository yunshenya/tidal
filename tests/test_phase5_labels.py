import numpy as np
from tidal.phase5_labels import (
    speech_interrupt, dyad_message_value, text_interjection, reply_message_value,
    domain_change_labels, nonspeech_span, split_hash,
)

def test_barge_in_positive_and_backchannel_negative():
    # A talks [0,2], B takes the floor at 1 and A yields (A ends, B continues, A stays quiet)
    y, kind = speech_interrupt([(0, 2, "A"), (1.0, 3.0, "B")])
    assert kind[1] == "barge" and y[1] == 1
    # short overlap, A continues past B
    y, kind = speech_interrupt([(0, 3, "A"), (1.0, 1.4, "B")])
    assert kind[1] == "bc" and y[1] == 0

def test_wait_negative_and_too_short_gap_masked():
    y, kind = speech_interrupt([(0, 1, "A"), (1.5, 2.5, "B")])
    assert kind[1] == "wait" and y[1] == 0
    y, kind = speech_interrupt([(0, 1, "A"), (1.1, 2.0, "B")])
    assert kind[1] == "mask" and np.isnan(y[1])

def test_resume_blocks_barge_and_long_overlap_is_masked():
    # A comes back within 0.40 s, so B did not take the floor
    y, kind = speech_interrupt([(0, 2, "A"), (1, 3, "B"), (2.2, 4, "A")])
    assert kind[1] == "mask" and np.isnan(y[1])
    # overlap longer than 1 s and A is still going when B ends -> not a backchannel, not a yield
    y, kind = speech_interrupt([(0, 3, "A"), (1, 2.5, "B")])
    assert kind[1] == "mask" and np.isnan(y[1])

def test_labels_ignore_text_and_filter_only_bracket_marks():
    assert nonspeech_span("none", "嗯")
    assert nonspeech_span("G0019", "[*]")
    assert nonspeech_span("G0019", "[+]")
    assert not nonspeech_span("G0019", "嗯")
    assert not nonspeech_span("G0019", "好的")
    a, _ = speech_interrupt([(0, 2, "A"), (1, 3, "B")])
    b, _ = speech_interrupt([(0, 2, "A"), (1, 3, "B")])
    assert np.array_equal(a, b, equal_nan=True)

def test_text_interjection_floor_take_vs_continue():
    # A, B inserts, A continues inside 8 s -> negative.
    y = text_interjection([0, 1, 3], ["A", "B", "A"])
    assert y[1] == 0 and np.isnan(y[0]) and np.isnan(y[2])
    # B inserts, keeps the floor, and A only returns after the 8 s gap -> positive.
    y = text_interjection([0, 1, 3, 20], ["A", "B", "B", "A"])
    assert y[1] == 1

def test_reply_message_value_uses_links_not_time_alone():
    ts = [0, 10, 80, 90]
    ids = ["m0", "m1", "m2", "m3"]
    reply = [None, "m0", None, None]
    y = reply_message_value(ts, ids, reply, minute_clock=False)
    assert y[0] == 1
    assert y[1] == 0          # 80 s left in the conversation, nobody points here
    assert np.isnan(y[3])     # conversation ends immediately
    y2 = reply_message_value(ts, ids, reply, minute_clock=True)
    assert y2[0] == 1 and y2[1] == 0 and np.isnan(y2[3])

def test_dyad_value_requires_non_backchannel_response():
    # other party answers with a clean turn inside 5 s
    segs = [(0, 1, "A"), (1.4, 3, "B")]
    y = dyad_message_value(segs)
    assert y[0] == 1
    assert np.isnan(y[1])  # B's response window is not observed (log ends at B)
    # backchannel does not count as message value
    segs = [(0, 3, "A"), (1.0, 1.3, "B"), (4.0, 9, "A")]
    y = dyad_message_value(segs)
    assert y[0] == 0  # next other segment is a backchannel; window to 8 s is observed (conv ends at 9)

def test_domain_change_is_set_difference_not_keywords():
    empty = {"hotel": {"name": ""}, "taxi": {"from": ""}}
    hotel = {"hotel": {"name": "x"}, "taxi": {"from": ""}}
    both = {"hotel": {"name": "x"}, "taxi": {"from": "a"}}
    y = domain_change_labels([hotel, hotel, both, empty])
    assert y.tolist() == [0, 0, 1, 1]
    y2 = domain_change_labels([None, hotel, both])
    assert np.isnan(y2[0]) and y2[1] == 0 and y2[2] == 1
    y3 = domain_change_labels([hotel, None, both])
    assert y3[0] == 0 and np.isnan(y3[1]) and y3[2] == 1

def test_split_hash_is_stable_and_three_ways():
    assert split_hash("abc") == split_hash("abc")
    got = {split_hash(f"c{i}") for i in range(200)}
    assert got <= {"pub_train", "pub_val", "pub_test"} and len(got) == 3
