# -*- coding: utf-8 -*-
"""autosrt_core birim testleri (ağ / GPU / model GEREKTİRMEZ).

    python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                     # noqa: E402

from autosrt_core import align as AL                   # noqa: E402
from autosrt_core import audio as AU                   # noqa: E402
from autosrt_core import judge as J                    # noqa: E402
from autosrt_core import parakeet as PK                # noqa: E402
from autosrt_core import pipeline as P                 # noqa: E402
from autosrt_core.jobs import Job, fingerprint, save_json  # noqa: E402


def W(text, s, e, seg_end=False):
    return {"word": " " + text, "start": s, "end": e, "seg_end": seg_end}


def lines_at(spans):
    return [{"id": i + 1, "start": a, "end": b, "whisper": "w%d" % (i + 1),
             "parakeet": "p%d" % (i + 1), "disagreement": False}
            for i, (a, b) in enumerate(spans)]


class TestValidate(unittest.TestCase):
    def test_ok(self):
        raw = json.dumps({"lines": [
            {"id": 1, "source_text": "Hi.", "turkish_text": "Selam.", "confidence": 0.9},
            {"id": 2, "source_text": "Bye.", "turkish_text": "Hoşça kal."}]})
        v = J.validate(raw, [1, 2])
        self.assertTrue(v.ok)
        self.assertEqual(v.good[1]["turkish_text"], "Selam.")
        self.assertIsNone(v.good[2]["confidence"])

    def test_missing_empty_extra_order(self):
        raw = {"lines": [
            {"id": 3, "source_text": "c", "turkish_text": "c"},
            {"id": 1, "source_text": "a", "turkish_text": ""},     # boş → eksik sayılır
            {"id": 9, "source_text": "x", "turkish_text": "x"}]}   # beklenmeyen
        v = J.validate(json.dumps(raw), [1, 2, 3])
        self.assertFalse(v.ok)
        self.assertEqual(v.missing, [1, 2])
        self.assertEqual(v.empty, [1])
        self.assertEqual(v.extra, [9])
        self.assertIn(3, v.good)

    def test_chat_answer_with_fence_and_chatter(self):
        txt = ("Tabii! İşte sonuç:\n```json\n" +
               json.dumps({"lines": [{"id": 5, "source_text": "s", "turkish_text": "t"}]}) +
               "\n```\nBaşka bir şey?")
        v = J.validate(txt, [5])
        self.assertTrue(v.ok)

    def test_truncated_answer_reports_parse_error(self):
        v = J.validate('{"lines": [{"id": 1, "source_text": "a", "turk', [1, 2])
        self.assertFalse(v.ok)
        self.assertIsNotNone(v.parse_error)
        self.assertEqual(v.missing, [1, 2])

    def test_bare_list_and_confidence_clamp(self):
        v = J.validate('[{"id": 1, "source_text": "a", "turkish_text": "b", "confidence": 7}]', [1])
        self.assertTrue(v.ok)
        self.assertEqual(v.good[1]["confidence"], 1.0)


class FakeErr(Exception):
    def __init__(self, code, status, details, msg="x"):
        super().__init__(msg)
        self.code, self.status, self.details, self.message = code, status, details, msg


class TestClassify(unittest.TestCase):
    def test_rate_with_retry_delay(self):
        e = FakeErr(429, "RESOURCE_EXHAUSTED", {"error": {"details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]}})
        je = J.classify(e)
        self.assertEqual(je.kind, "rate")
        self.assertEqual(je.retry_after, 37.0)

    def test_daily(self):
        e = FakeErr(429, "RESOURCE_EXHAUSTED", {"error": {"details": [
            {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
        self.assertEqual(J.classify(e).kind, "quota_daily")

    def test_auth_model_server_net(self):
        self.assertEqual(J.classify(FakeErr(400, "INVALID_ARGUMENT", {}, "API key not valid")).kind, "auth")
        self.assertEqual(J.classify(FakeErr(404, "NOT_FOUND", {}, "models/x is not found")).kind, "model")
        self.assertEqual(J.classify(FakeErr(503, "UNAVAILABLE", {})).kind, "server")
        self.assertEqual(J.classify(ConnectionError("connection reset")).kind, "net")

    def test_key_is_masked(self):
        je = J.classify(FakeErr(400, "X", {}, "bad AIzaSyA1234567890abcdefghijklmnop key"))
        self.assertNotIn("AIzaSyA1234567890", je.detail)


class TestPlanChunks(unittest.TestCase):
    def test_respects_limit_and_never_splits_a_line(self):
        spans, t = [], 0.0
        for i in range(400):                       # ~20 dk, her 3 sn bir satır
            spans.append((t, t + 2.0 + (1.5 if i % 7 == 0 else 0)))
            t += 3.0 + (2.0 if i % 13 == 0 else 0)
        lines = lines_at(spans)
        chunks = P.plan_chunks(lines, {ln["id"] for ln in lines}, 300)
        self.assertEqual(sum(len(c) for c in chunks), len(lines))
        ids = [ln["id"] for c in chunks for ln in c]
        self.assertEqual(ids, sorted(ids))
        for c in chunks:
            self.assertLessEqual(c[-1]["end"] - c[0]["start"], 300)
        self.assertGreaterEqual(len(chunks), 5)

    def test_prefers_longest_silence(self):
        spans = [(0, 50), (51, 100), (110, 150), (151, 200), (201, 240)]   # 10 sn boşluk 100–110
        chunks = P.plan_chunks(lines_at(spans), {1, 2, 3, 4, 5}, 160)
        self.assertEqual([ln["id"] for ln in chunks[0]], [1, 2])

    def test_only_pending(self):
        lines = lines_at([(0, 1), (2, 3), (4, 5), (6, 7)])
        chunks = P.plan_chunks(lines, {2, 4}, 600)
        self.assertEqual([[ln["id"] for ln in c] for c in chunks], [[2, 4]])


class TestOutputs(unittest.TestCase):
    def test_timestamps_from_whisper_and_fallback(self):
        lines = lines_at([(1.0, 2.0), (3.0, 4.0)])
        lines[0]["disagreement"] = True
        res = {1: {"source_text": "Hello.", "turkish_text": "Merhaba.", "confidence": 0.4, "via": "auto"},
               2: {"source_text": "w2", "turkish_text": "w2", "via": "fallback"}}
        src, tr, flags = P.build_outputs(lines, res, "hybrid")
        self.assertEqual(src[0], [1.0, 2.0, "Hello."])
        self.assertEqual(tr[0], [1.0, 2.0, "Merhaba."])
        self.assertTrue(flags[0]["dis"] and flags[0]["low"])
        self.assertTrue(flags[1]["unverified"])
        self.assertEqual(tr[1][2], "w2")

    def test_whisper_parakeet_mode_has_no_turkish(self):
        src, tr, flags = P.build_outputs(lines_at([(0, 1)]), {}, "whisper_parakeet")
        self.assertIsNone(tr)
        self.assertEqual(src[0][2], "w1")

    def test_min_duration_same_on_both(self):
        lines = lines_at([(0.0, 0.3), (2.0, 2.5)])
        res = {1: {"source_text": "a", "turkish_text": "b"}, 2: {"source_text": "c", "turkish_text": "d"}}
        src, tr, _ = P.build_outputs(lines, res, "hybrid", min_dur=1.0)
        self.assertAlmostEqual(src[0][1], 1.0)
        self.assertEqual([b[:2] for b in src], [b[:2] for b in tr])


class TestAlign(unittest.TestCase):
    def test_owner_mapping_and_flags(self):
        ww = [W("Hello", 0.0, 0.4), W("world.", 0.5, 0.9), W("Komodo", 2.0, 2.5, True),
              W("dragon", 2.6, 3.0, True)]
        blocks = [[0.0, 0.9, "Hello world."], [2.0, 2.5, "Komodo"], [2.6, 3.0, "dragon"]]
        pw = [W("hello", 0.05, 0.4), W("world!", 0.5, 0.9), W("Comodo", 2.0, 2.5),
              W("dragon", 2.6, 3.0)]
        lines, st = AL.align(ww, pw, blocks, "en")
        self.assertEqual([ln["disagreement"] for ln in lines], [False, True, False])
        self.assertEqual(lines[1]["parakeet"], "Comodo")
        self.assertEqual(st["orphans"], 0)

    def test_no_parakeet(self):
        ww = [W("Hi", 0, 1)]
        lines, _ = AL.align(ww, [], [[0, 1, "Hi"]], "en")
        self.assertIsNone(lines[0]["parakeet"])
        self.assertFalse(lines[0]["disagreement"])


class TestParakeetTokens(unittest.TestCase):
    def test_tokens_to_words(self):
        toks = [" out", " of", " ac", "tion", ".", " H", "is"]
        ts = [0.24, 0.32, 0.48, 0.56, 1.12, 1.28, 1.36]
        w = PK.tokens_to_words(toks, ts, 10.0)
        self.assertEqual([x["word"] for x in w], [" out", " of", " action.", " His"])
        self.assertAlmostEqual(w[2]["start"], 10.48)
        self.assertTrue(all(x["end"] > x["start"] for x in w))

    def test_supports(self):
        for c in ("en", "ru", "fr", "es", "de"):
            self.assertTrue(PK.supports(c))
        self.assertFalse(PK.supports("tr"))


class TestJobs(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.video = os.path.join(self.root, "v.mp4")
        with open(self.video, "wb") as f:
            f.write(b"x")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_dependency_graph(self):
        j = Job(self.video, root=os.path.join(self.root, "jobs"))
        for s in ("audio", "whisper", "parakeet", "lines"):
            save_json(j.path(s + ".json"), {})
            j.mark_done(s, "fp", file=s + ".json")
        j.save_results({1: {"source_text": "a"}})
        j.invalidate("whisper")               # Whisper ayarı değişti
        self.assertTrue(j.is_done("parakeet", "fp"))      # Parakeet korunur
        self.assertTrue(j.is_done("audio", "fp"))
        self.assertFalse(j.is_done("whisper", "fp"))
        self.assertFalse(j.is_done("lines", "fp"))
        self.assertEqual(j.load_results(), {})            # satır id'leri değişti → sonuçlar gider

    def test_same_video_same_job_and_cleanup(self):
        root = os.path.join(self.root, "jobs")
        a, b = Job(self.video, root=root), Job(self.video, root=root)
        self.assertEqual(a.dir, b.dir)
        a.merge_results({1: {"x": 1}})
        self.assertEqual(Job(self.video, root=root).load_results(), {1: {"x": 1}})
        a.cleanup()
        self.assertFalse(os.path.exists(a.dir))

    def test_fingerprint_stable(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))


class TestManualSession(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.video = os.path.join(self.root, "v.mp4")
        with open(self.video, "wb") as f:
            f.write(b"x")
        self.jr = os.path.join(self.root, "jobs")
        j = Job(self.video, root=self.jr)
        AU.write_wav(np.zeros(16000 * 20, dtype=np.float32), j.path("audio.wav"))
        j.mark_done("audio", "fp", file="audio.wav", duration=20.0)
        save_json(j.path("lines.json"), {"lines": lines_at([(1, 2), (3, 4), (5, 6)]),
                                         "src_lang": "en"})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_partial_answer_then_rest(self):
        ms = P.ManualSession(self.video, chunk_min=5, jobs_root=self.jr)
        pkg = ms.next_package()
        self.assertEqual(pkg["ids"], [1, 2, 3])
        self.assertTrue(os.path.exists(pkg["wav"]))
        self.assertIn('"disagreement"', pkg["prompt"])
        self.assertIn("NEVER change ids", pkg["prompt"])
        v = ms.submit('```json\n{"lines":[{"id":1,"source_text":"a","turkish_text":"b"},'
                      '{"id":3,"source_text":"c","turkish_text":"d"}]}\n```')
        self.assertEqual(v.missing, [2])
        pkg2 = ms.next_package()                # eksik id bir sonraki pakete girer
        self.assertEqual(pkg2["ids"], [2])
        ms.submit('{"lines":[{"id":2,"source_text":"x","turkish_text":"y"}]}')
        self.assertTrue(ms.all_done())
        self.assertIsNone(ms.next_package())
        res = ms.job.load_results()
        self.assertEqual(res[2]["via"], "manual")


class TestAudio(unittest.TestCase):
    def test_speech_windows(self):
        sr = 16000
        chunks = [{"start": 0, "end": 5 * sr}, {"start": 6 * sr, "end": 10 * sr},
                  {"start": 30 * sr, "end": 90 * sr}]
        w = AU.speech_windows(chunks, 100.0, max_len=24.0)
        self.assertEqual(w[0], (0.0, 10.0))           # yakın parçalar birleşti
        self.assertTrue(all(b - a <= 24.0 + 1e-9 for a, b in w))
        self.assertAlmostEqual(sum(b - a for a, b in w[1:]), 60.0)


if __name__ == "__main__":
    unittest.main()
