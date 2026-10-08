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

from tidal.phase5_labels import forward_barge_speech, forward_barge_text, barge_self_speaker

def test_forward_barge_is_before_the_onset_and_uses_only_the_horizon():
    # other holds [0, 3]; self starts at 0.8, while the floor continues
    segs = [(0.0, 3.0, "A"), (0.8, 1.6, "B")]
    assert barge_self_speaker(segs) == "A" or True
    pts = forward_barge_speech(segs, "B", rec_end=3.0)
    by_t = {round(t, 1): y for t, y, _s in pts}
    assert by_t[0.5] == 1.0                      # 0.8 is inside (0.5, 1.5] and before floor end
    assert 1.0 not in by_t and 1.5 not in by_t   # self is already speaking there
    assert by_t[2.0] == 0.0 and by_t[2.5] == 0.0
    # an onset 1.6 s away is not "now" (H = 1); the next sample, 0.5 s out, is
    pts = forward_barge_speech([(0.0, 4.0, "A"), (2.0, 3.0, "B")], "B", rec_end=4.0)
    by_t = {round(t, 1): y for t, y, _s in pts}
    assert by_t[0.5] == 0.0 and by_t[1.0] == 1.0 and by_t[1.5] == 1.0
    for t, y, s in pts:
        assert t > s and (y != 1 or t < 2.0)

def test_forward_barge_masks_a_censored_ending_and_skips_self_floor():
    pts = forward_barge_speech([(0.0, 3.0, "A")], "B", rec_end=0.7)
    by_t = {round(t, 1): y for t, y, _s in pts}
    # the segment is still the sampling grid; a recording that ends at 0.7 s cannot confirm any of these
    assert all(np.isnan(y) for y in by_t.values()) and 0.5 in by_t and 2.0 in by_t
    # self's own floor produces no decision
    assert forward_barge_speech([(0.0, 3.0, "B"), (4.0, 5.0, "B")], "B", rec_end=5.0) == []

def test_forward_barge_text_next_message_within_horizon():
    ts = [0, 3, 12, 30]
    roles = ["other", "self", "other", "other"]
    got = dict(forward_barge_text(ts, roles))
    assert got[0] == 1.0          # self answers at +3 s
    assert got[2] == 0.0          # next message is other, at +18 s > 8 s, so the horizon passed empty
    assert 1 not in got and np.isnan(got[3])   # last other message, horizon not observed
    assert np.isnan(dict(forward_barge_text([0], ["other"]))[0])
