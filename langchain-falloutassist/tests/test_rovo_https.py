"""Canonical HTTPS origin and session-cookie behavior for Rovo OAuth."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from falloutassist_next import rovo_a2a, web


class RovoHttpsTests(unittest.TestCase):
    def test_configured_https_origin_drives_callback_and_secure_cookie(self):
        with patch.dict(os.environ, {"ROVO_PUBLIC_BASE_URL": "https://falloutassist.example.com",
                                     "SESSION_COOKIE_SECURE": ""}, clear=True):
            base = web._rovo_public_base_url(SimpleNamespace(base_url="http://127.0.0.1:8012/"))
            self.assertEqual(base, "https://falloutassist.example.com")
            self.assertEqual(rovo_a2a.callback_url(base),
                             "https://falloutassist.example.com/api/rovo-a2a/callback")
            self.assertTrue(web._secure_session_cookie())

    def test_localhost_fallback_and_explicit_local_cookie_override(self):
        with patch.dict(os.environ, {"SESSION_COOKIE_SECURE": "false"}, clear=True):
            self.assertEqual(web._rovo_public_base_url(
                SimpleNamespace(base_url="http://127.0.0.1:8012/")), "http://127.0.0.1:8012/")
            self.assertEqual(rovo_a2a.callback_url("http://127.0.0.1:8012/"),
                             "http://127.0.0.1:8012/api/rovo-a2a/callback")
            self.assertFalse(web._secure_session_cookie())

    def test_explicit_secure_cookie_can_be_forced_without_public_origin(self):
        with patch.dict(os.environ, {"SESSION_COOKIE_SECURE": "true"}, clear=True):
            self.assertTrue(web._secure_session_cookie())


if __name__ == "__main__":
    unittest.main()
