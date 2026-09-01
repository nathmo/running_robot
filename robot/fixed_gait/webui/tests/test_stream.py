"""The single-connection UI transport: /api/stream and uiproc's feed.

WHY IT EXISTS. Measured on the robot 2026-09-01, idle and LIMP: the 200 Hz control loop holds
193 Hz with nothing polling and 107 Hz with one browser tab open, and the cost tracks REQUEST COUNT
rather than payload -- about 1 Hz of control rate per upstream request per second, whichever
endpoint it is. The 2026-08-19 process split moved the JSON work out but left ~22 upstream requests
a second. This replaces them with one connection.

Two properties have to hold or the change is a regression rather than an optimisation:

  * the cursor contract survives. The stream hands each sample over exactly ONCE, so uiproc has to
    keep local history; serving `since` from the newest frame alone would silently drop samples
    for any client polling slower than the frame rate.
  * losing the stream loses the optimisation and NOTHING else. Upstream restarts, this process
    starting first, an older server.py with no /api/stream -- all of them must fall back to the
    request-per-poll path that shipped before.

    python -m pytest robot/fixed_gait/webui/tests/test_stream.py -v
"""
import json
import struct

import numpy as np
import pytest

import paths
import uiproc


def _frame(seq, n, fields=("pos_raw", "pos_norm"), n_motors=None):
    n_motors = n_motors or paths.N_MOTORS
    t = np.arange(n, dtype=np.float64) + seq
    arr = np.zeros((len(fields), n, n_motors), np.float32) + seq
    return seq, list(fields), t, arr


class TestAppend:
    """uiproc._append -- the local history that makes a `since` cursor answerable."""

    def test_frames_accumulate_in_order(self):
        prev = None
        for seq, n in ((10, 10), (20, 10), (30, 10)):
            s, f, t, a = _frame(seq, n)
            prev = uiproc._append(prev, s, f, t, a)
        assert prev[0] == 30
        assert prev[2].size == 30
        assert prev[3].shape == (2, 30, paths.N_MOTORS)

    def test_a_frame_with_no_new_samples_keeps_the_history(self):
        """The common case at 20 Hz: the ring often has nothing new between frames. That must not
        erase what we already hold, or every quiet moment would blank the charts."""
        s, f, t, a = _frame(10, 10)
        prev = uiproc._append(None, s, f, t, a)
        empty_t = np.zeros(0, np.float64)
        empty_a = np.zeros((2, 0, paths.N_MOTORS), np.float32)
        prev = uiproc._append(prev, 10, f, empty_t, empty_a)
        assert prev[2].size == 10 and prev[3].shape[1] == 10

    def test_history_is_bounded(self):
        prev = None
        for k in range(40):
            s, f, t, a = _frame((k + 1) * 100, 100)
            prev = uiproc._append(prev, s, f, t, a)
        assert prev[2].size == uiproc.FEED_KEEP
        assert prev[3].shape[1] == uiproc.FEED_KEEP

    def test_a_field_change_restarts_rather_than_misaligning(self):
        """If upstream's field list ever changes, concatenating would silently put one field's
        samples under another field's name."""
        s, f, t, a = _frame(10, 10)
        prev = uiproc._append(None, s, f, t, a)
        s2, f2, t2, a2 = _frame(20, 10, fields=("pos_raw", "pos_norm", "spd"))
        prev = uiproc._append(prev, s2, f2, t2, a2)
        assert prev[1] == list(f2) and prev[3].shape[0] == 3 and prev[2].size == 10


class TestCursor:
    """FEED.since -- what a browser polling at its own rate actually receives."""

    @pytest.fixture
    def feed(self):
        f = uiproc._Feed()
        f.connected = True
        s, fl, t, a = _frame(0, 0)
        f.tel = uiproc._append(None, 100, ["pos_raw", "pos_norm"],
                               np.arange(100, dtype=np.float64),
                               np.zeros((2, 100, paths.N_MOTORS), np.float32))
        return f

    def test_a_cursor_asks_for_exactly_what_it_has_not_seen(self, feed):
        seq, _f, t, arr, _m = feed.since("tel", 80)
        assert seq == 100
        assert t.size == 20 and arr.shape[1] == 20, (
            "a client 20 samples behind must get 20 -- not one frame's worth, which is the bug "
            "serving `since` from the newest frame alone would have")

    def test_an_up_to_date_cursor_gets_nothing(self, feed):
        seq, _f, t, arr, _m = feed.since("tel", 100)
        assert seq == 100 and t.size == 0 and arr.shape[1] == 0

    def test_a_cursor_older_than_the_history_gets_the_whole_history(self, feed):
        _seq, _f, t, _arr, _m = feed.since("tel", 0)
        assert t.size == 100

    def test_a_nonsense_cursor_is_not_a_crash(self, feed):
        """A fresh page sends `since=undefined` before it has one."""
        for bad in (None, "undefined", "", "abc"):
            seq, _f, t, _a, _m = feed.since("tel", bad)
            assert seq == 100 and t.size == 100

    def test_nothing_is_served_while_the_stream_is_down(self, feed):
        """None is the signal every handler uses to take the fallback path. Serving stale frames
        instead would show a frozen robot as a live one."""
        feed.connected = False
        assert feed.since("tel", 0) is None


def test_the_frame_format_round_trips():
    """Header length, header JSON, then raw bytes -- the reader and the writer must agree byte for
    byte, because a one-field drift desynchronises the stream permanently rather than erroring."""
    import server
    t = np.arange(5, dtype=np.float64)
    arr = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
    blob = server._pack_frame({"tel": {"n": 5, "shape": [2, 5, 6]}}, [t, arr])
    n = struct.unpack(">I", blob[:4])[0]
    header = json.loads(blob[4:4 + n].decode())
    body = blob[4 + n:]
    assert header["tel"]["shape"] == [2, 5, 6]
    got_t = np.frombuffer(body[:t.nbytes], np.float64)
    got_a = np.frombuffer(body[t.nbytes:t.nbytes + arr.nbytes], np.float32).reshape(2, 5, 6)
    assert np.array_equal(got_t, t) and np.array_equal(got_a, arr)
    assert len(body) == t.nbytes + arr.nbytes, "no padding, no trailing bytes"
