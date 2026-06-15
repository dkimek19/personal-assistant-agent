"""
Runnable test for session_id_resolver (Sub-AC 6.2).

Verifies that the same logical user arriving via each of the three interfaces
(web_ui, telegram, discord) resolves to the *same* canonical user_id and the
*same* stable session_id — the core requirement of unified cross-interface
session identity.

Run with:
    python -m pytest tests/test_session_id_resolver.py -v
    # — or — without pytest:
    python tests/test_session_id_resolver.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the package root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.session_store import SessionStore
from assistant.session_resolver import (
    CANONICAL_USER_ID,
    VALID_INTERFACES,
    ResolvedSession,
    session_id_resolver,
)


class TestSessionIdResolver(unittest.TestCase):
    """Tests for session_id_resolver — Sub-AC 6.2."""

    # ------------------------------------------------------------------
    # Test fixtures
    # ------------------------------------------------------------------

    def setUp(self) -> None:
        """Each test gets its own isolated temp database."""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = SessionStore(db_path=self._tmp.name)

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)
        for ext in ("-wal", "-shm"):
            Path(self._tmp.name + ext).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Core AC 6.2 assertion
    # ------------------------------------------------------------------

    def test_all_interfaces_resolve_to_same_session_id(self) -> None:
        """
        **Primary AC 6.2 assertion.**

        The same logical user arriving via web_ui, telegram, and discord
        must all resolve to the *same* canonical session_id.
        """
        result_web = session_id_resolver("web_ui", store=self.store)
        result_telegram = session_id_resolver("telegram", store=self.store)
        result_discord = session_id_resolver("discord", store=self.store)

        self.assertEqual(
            result_web.session_id,
            result_telegram.session_id,
            "web_ui and telegram must share the same session_id",
        )
        self.assertEqual(
            result_web.session_id,
            result_discord.session_id,
            "web_ui and discord must share the same session_id",
        )
        self.assertEqual(
            result_telegram.session_id,
            result_discord.session_id,
            "telegram and discord must share the same session_id",
        )

    def test_all_interfaces_resolve_to_canonical_user_id(self) -> None:
        """Every interface must map to the single canonical user_id."""
        for interface in sorted(VALID_INTERFACES):
            with self.subTest(interface=interface):
                result = session_id_resolver(interface, store=self.store)
                self.assertEqual(
                    result.user_id,
                    CANONICAL_USER_ID,
                    f"Interface {interface!r} must resolve to CANONICAL_USER_ID",
                )

    # ------------------------------------------------------------------
    # Return-type correctness
    # ------------------------------------------------------------------

    def test_returns_resolved_session_namedtuple(self) -> None:
        """Return value must be a ResolvedSession namedtuple."""
        result = session_id_resolver("web_ui", store=self.store)
        self.assertIsInstance(result, ResolvedSession)
        self.assertTrue(hasattr(result, "user_id"))
        self.assertTrue(hasattr(result, "session_id"))

    def test_session_id_is_uuid4_format(self) -> None:
        """The returned session_id must be a 36-character UUID4 string."""
        result = session_id_resolver("telegram", store=self.store)

        self.assertIsInstance(result.session_id, str)
        self.assertEqual(len(result.session_id), 36, "UUID4 must be 36 characters")

        parts = result.session_id.split("-")
        self.assertEqual(len(parts), 5, "UUID4 must have 5 hyphen-separated groups")
        self.assertEqual(
            parts[2][0], "4", "Third group's first nibble must be '4' for UUID version 4"
        )

    # ------------------------------------------------------------------
    # Stability & idempotency
    # ------------------------------------------------------------------

    def test_session_id_stable_across_repeated_calls_same_interface(self) -> None:
        """Repeated calls from the same interface must return the same session_id."""
        r1 = session_id_resolver("telegram", store=self.store)
        r2 = session_id_resolver("telegram", store=self.store)
        r3 = session_id_resolver("telegram", store=self.store)
        self.assertEqual(r1.session_id, r2.session_id)
        self.assertEqual(r2.session_id, r3.session_id)

    def test_session_id_stable_across_repeated_calls_mixed_interfaces(self) -> None:
        """Mixed-interface calls interleaved must all return the same session_id."""
        ids = [
            session_id_resolver(iface, store=self.store).session_id
            for iface in ["web_ui", "telegram", "discord", "web_ui", "telegram"]
        ]
        self.assertEqual(len(set(ids)), 1, "All calls must yield the same session_id")

    # ------------------------------------------------------------------
    # Forward-compat: interface_user_id parameter
    # ------------------------------------------------------------------

    def test_interface_user_id_accepted_without_error(self) -> None:
        """interface_user_id is accepted and does not raise."""
        r = session_id_resolver(
            "telegram", interface_user_id="123456789", store=self.store
        )
        self.assertEqual(r.user_id, CANONICAL_USER_ID)

    def test_interface_user_id_does_not_change_canonical_user(self) -> None:
        """
        Different interface_user_id values across interfaces must not
        split the session — all still resolve to the same session_id.
        """
        r_tg = session_id_resolver(
            "telegram", interface_user_id="TG_CHAT_111", store=self.store
        )
        r_dc = session_id_resolver(
            "discord", interface_user_id="DC_USER_222", store=self.store
        )
        r_web = session_id_resolver(
            "web_ui", interface_user_id="WEB_CLIENT_333", store=self.store
        )
        self.assertEqual(r_tg.session_id, r_dc.session_id)
        self.assertEqual(r_dc.session_id, r_web.session_id)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_unknown_interface_raises_value_error(self) -> None:
        """An unrecognised interface name must raise ValueError."""
        with self.assertRaises(ValueError):
            session_id_resolver("sms", store=self.store)

    def test_empty_interface_raises_value_error(self) -> None:
        """An empty string interface name must raise ValueError."""
        with self.assertRaises(ValueError):
            session_id_resolver("", store=self.store)

    def test_interface_case_sensitive(self) -> None:
        """Interface names are case-sensitive; 'Telegram' is not 'telegram'."""
        with self.assertRaises(ValueError):
            session_id_resolver("Telegram", store=self.store)

    # ------------------------------------------------------------------
    # Single-user store integrity
    # ------------------------------------------------------------------

    def test_single_session_in_store_after_all_interfaces(self) -> None:
        """
        After resolving via all three interfaces, the store must contain
        exactly one session — the single-user invariant.
        """
        session_id_resolver("web_ui", store=self.store)
        session_id_resolver("telegram", store=self.store)
        session_id_resolver("discord", store=self.store)

        users = self.store.list_users()
        self.assertEqual(
            len(users),
            1,
            f"Single-user system: expected 1 session, found {len(users)}: {users}",
        )
        self.assertEqual(users[0], CANONICAL_USER_ID)

    def test_session_id_matches_store_session_id(self) -> None:
        """
        The session_id returned by session_id_resolver must match the
        session_id stored in the SessionStore for the canonical user.
        """
        result = session_id_resolver("discord", store=self.store)
        stored = self.store.get_session(CANONICAL_USER_ID)

        self.assertIsNotNone(stored)
        self.assertEqual(result.session_id, stored["_meta"]["session_id"])

    def test_bootstrap_context_contains_expected_keys(self) -> None:
        """
        When a session is bootstrapped by the resolver, the initial context
        must include the required ontology keys.
        """
        session_id_resolver("web_ui", store=self.store)
        ctx = self.store.get_session(CANONICAL_USER_ID)

        self.assertIsNotNone(ctx)
        self.assertIn("working_memory", ctx)
        self.assertIn("session_memory", ctx)
        self.assertIn("long_term_memory", ctx)
        self.assertIn("source_interface", ctx)

    def test_existing_session_not_overwritten(self) -> None:
        """
        If the canonical session already exists (e.g. bootstrapped by web_ui),
        a subsequent call from telegram must not reset or overwrite the context.
        """
        # Bootstrap via web_ui and add some working memory
        self.store.upsert_session(
            CANONICAL_USER_ID,
            {
                "working_memory": [{"role": "user", "content": "Hello from web"}],
                "source_interface": "web_ui",
                "session_memory": {},
                "long_term_memory": [],
            },
        )

        # Now resolve via telegram — must not wipe existing session
        session_id_resolver("telegram", store=self.store)

        ctx = self.store.get_session(CANONICAL_USER_ID)
        self.assertIsNotNone(ctx)
        self.assertEqual(
            len(ctx["working_memory"]),
            1,
            "Existing working_memory must be preserved when resolver is called on an existing session",
        )


# ---------------------------------------------------------------------------
# Allow running the test file directly (without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSessionIdResolver)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
