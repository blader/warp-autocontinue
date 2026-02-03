import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import warp_autocontinue as wa


def _encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return bytes(out)


def _make_uuid_framed_message(*, uuid: str, role: str, text: str) -> bytes:
    assert len(uuid.encode("ascii")) == wa.UUID_ASCII_LEN
    tag = (
        wa.TASK_BLOB_TAG_USER_TEXT
        if role == "user"
        else wa.TASK_BLOB_TAG_ASSISTANT_TEXT
    )
    text_bytes = text.encode("utf-8")
    inner = bytes([wa.TASK_BLOB_TAG_STRING]) + _encode_varint(len(text_bytes)) + text_bytes
    return uuid.encode("ascii") + bytes([tag]) + _encode_varint(len(inner)) + inner


class TestVarint(unittest.TestCase):
    def test_read_varint_simple(self) -> None:
        buf = _encode_varint(300) + b"xyz"
        val, next_i = wa.read_varint(buf, 0)
        self.assertEqual(val, 300)
        self.assertEqual(next_i, len(_encode_varint(300)))

    def test_read_varint_raises_on_eof(self) -> None:
        # 0x80 indicates continuation, but the buffer ends.
        with self.assertRaises(ValueError):
            wa.read_varint(b"\x80", 0)


class TestMessageExtraction(unittest.TestCase):
    def test_parse_uuid_framed_message_user(self) -> None:
        uuid = "123e4567-e89b-12d3-a456-426614174000"
        msg = _make_uuid_framed_message(uuid=uuid, role="user", text="hello")
        parsed = wa._parse_uuid_framed_message_at(msg, 0)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.uuid, uuid)
        self.assertEqual(parsed.role, "user")
        self.assertEqual(parsed.text, "hello")

    def test_parse_uuid_framed_message_assistant(self) -> None:
        uuid = "123e4567-e89b-12d3-a456-426614174001"
        msg = _make_uuid_framed_message(uuid=uuid, role="assistant", text="hi")
        parsed = wa._parse_uuid_framed_message_at(msg, 0)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.role, "assistant")
        self.assertEqual(parsed.text, "hi")

    def test_extract_messages_sorts_by_offset(self) -> None:
        uuid1 = "123e4567-e89b-12d3-a456-426614174010"
        uuid2 = "123e4567-e89b-12d3-a456-426614174011"
        m1 = _make_uuid_framed_message(uuid=uuid1, role="user", text="one")
        m2 = _make_uuid_framed_message(uuid=uuid2, role="assistant", text="two")

        blob = b"noise" + m2 + b"more-noise" + m1
        msgs = wa.extract_messages_from_agent_task_blob(blob)
        self.assertEqual([m.uuid for m in msgs], [uuid2, uuid1])


class TestHelpers(unittest.TestCase):
    def test_get_recent_user_messages(self) -> None:
        msgs = [
            wa.ExtractedMessage(0, "u1", "user", "A"),
            wa.ExtractedMessage(1, "a1", "assistant", "..."),
            wa.ExtractedMessage(2, "u2", "user", "B"),
            wa.ExtractedMessage(3, "u3", "user", "C"),
        ]
        self.assertEqual(wa.get_recent_user_messages(msgs, 2), ["B", "C"])

    def test_heuristic_should_continue_errors_and_questions(self) -> None:
        ok, _ = wa.heuristic_should_continue(assistant_text="Traceback (most recent call last):")
        self.assertTrue(ok)

        ok, _ = wa.heuristic_should_continue(assistant_text="Should I proceed?")
        self.assertTrue(ok)

    def test_heuristic_should_continue_done(self) -> None:
        ok, _ = wa.heuristic_should_continue(assistant_text="All set. Finished.")
        self.assertFalse(ok)

    def test_build_eval_prompt_truncates_plan(self) -> None:
        plan = wa.PlanDoc(
            notebook_id=1,
            ai_document_id="doc",
            title="My Plan",
            markdown="x" * 50,
        )
        prompt = wa.build_eval_prompt(
            user_messages=["u"],
            assistant_message="a",
            plan=plan,
            max_plan_chars=10,
        )
        self.assertIn("In-context Plan: My Plan", prompt)
        self.assertIn("…(truncated)…", prompt)
        self.assertIn("Last assistant response:", prompt)


class TestDbLocation(unittest.TestCase):
    def test_locate_db_prefers_preview_then_stable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)

            app_support = home / wa.WARP_APP_SUPPORT_SUBPATH
            preview = app_support / wa.WARP_DB_PREVIEW_SUBPATH
            stable = app_support / wa.WARP_DB_STABLE_SUBPATH

            preview.parent.mkdir(parents=True, exist_ok=True)
            stable.parent.mkdir(parents=True, exist_ok=True)

            stable.write_text("")
            preview.write_text("")

            with mock.patch("pathlib.Path.home", return_value=home):
                self.assertEqual(wa.locate_warp_db_path(None), preview)

            preview.unlink()
            with mock.patch("pathlib.Path.home", return_value=home):
                self.assertEqual(wa.locate_warp_db_path(None), stable)

    def test_locate_db_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "warp.sqlite"
            p.write_text("")
            self.assertEqual(wa.locate_warp_db_path(str(p)), p)


class TestPlanExtraction(unittest.TestCase):
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE ai_document_panes (
              id INTEGER PRIMARY KEY,
              document_id TEXT NOT NULL,
              version INTEGER NOT NULL
            );
            CREATE TABLE notebooks (
              id INTEGER PRIMARY KEY,
              title TEXT,
              data TEXT,
              ai_document_id TEXT
            );
            """
        )
        return conn

    def test_get_latest_plan_doc_for_blob(self) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO ai_document_panes (id, document_id, version) VALUES (1, ?, 1)",
            ("doc-1",),
        )
        conn.execute(
            "INSERT INTO notebooks (id, title, data, ai_document_id) VALUES (1, ?, ?, ?)",
            ("Plan", "# hi", "doc-1"),
        )
        blob = b"prefix doc-1 suffix"
        plan = wa.get_latest_plan_doc_for_blob(conn, blob)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.title, "Plan")
        self.assertEqual(plan.markdown, "# hi")

    def test_get_latest_plan_doc_prefers_newest_pane(self) -> None:
        conn = self._conn()
        # Older
        conn.execute(
            "INSERT INTO ai_document_panes (id, document_id, version) VALUES (1, ?, 1)",
            ("doc-old",),
        )
        conn.execute(
            "INSERT INTO notebooks (id, title, data, ai_document_id) VALUES (1, ?, ?, ?)",
            ("Old", "old", "doc-old"),
        )
        # Newer
        conn.execute(
            "INSERT INTO ai_document_panes (id, document_id, version) VALUES (2, ?, 1)",
            ("doc-new",),
        )
        conn.execute(
            "INSERT INTO notebooks (id, title, data, ai_document_id) VALUES (2, ?, ?, ?)",
            ("New", "new", "doc-new"),
        )

        blob = b"doc-old ... doc-new"
        plan = wa.get_latest_plan_doc_for_blob(conn, blob)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.title, "New")

    def test_get_latest_plan_doc_skips_missing_notebook(self) -> None:
        conn = self._conn()
        # Newest pane has no notebook row.
        conn.execute(
            "INSERT INTO ai_document_panes (id, document_id, version) VALUES (2, ?, 1)",
            ("doc-missing",),
        )
        # Older pane does.
        conn.execute(
            "INSERT INTO ai_document_panes (id, document_id, version) VALUES (1, ?, 1)",
            ("doc-ok",),
        )
        conn.execute(
            "INSERT INTO notebooks (id, title, data, ai_document_id) VALUES (1, ?, ?, ?)",
            ("OK", "ok", "doc-ok"),
        )

        blob = b"doc-missing doc-ok"
        plan = wa.get_latest_plan_doc_for_blob(conn, blob)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.title, "OK")


if __name__ == "__main__":
    unittest.main()
