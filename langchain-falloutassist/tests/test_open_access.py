"""The UI has no account login; service-to-service keys and CSRF remain enforced."""

import os
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

from falloutassist_next import storage


class OpenAccessTests(unittest.TestCase):
    def setUp(self):
        original = storage.connect
        path = Path.cwd() / f".open-access-{uuid.uuid4().hex}.sqlite3"

        def cleanup_database():
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(path) + suffix)
                if candidate.exists():
                    candidate.unlink()

        self.addCleanup(cleanup_database)
        for item in (
            patch.object(storage, "database_backend", return_value="sqlite"),
            patch.object(storage, "connect", side_effect=lambda: original(path)),
            patch.dict(os.environ, {"SESSION_SECRET": "isolated-open-access-test-secret"}),
        ):
            item.start()
            self.addCleanup(item.stop)
        storage.initialize()

    def test_dashboard_is_open_without_login_and_auth_routes_are_gone(self):
        from fastapi.testclient import TestClient
        from falloutassist_next import web

        with TestClient(web.app, base_url="http://192.0.2.50:8012") as client:
            session = client.get("/api/session")
            self.assertEqual(session.status_code, 200)
            self.assertEqual(session.json()["user"]["role"], "admin")
            self.assertEqual(client.get("/api/dashboard").status_code, 200)
            self.assertEqual(client.get("/api/connections").status_code, 200)
            self.assertEqual(client.get("/api/auth/me").status_code, 404)
            self.assertEqual(client.post("/api/auth/login", json={}).status_code, 404)
            self.assertEqual(client.post("/api/auth/bootstrap", json={}).status_code, 404)
            self.assertEqual(client.post("/api/auth/logout").status_code, 404)
            page = client.get("/").text
            self.assertNotIn('id="login"', page)
            self.assertNotIn('id="firstAdminSetup"', page)
            self.assertIn("No FalloutAssist sign-in is required", page)

    def test_csrf_and_downstream_agent_key_still_apply(self):
        from fastapi.testclient import TestClient
        from falloutassist_next import web

        with patch.object(web, "config_value", return_value=""), \
                TestClient(web.app, base_url="http://192.0.2.50:8012") as client:
            self.assertEqual(client.post("/api/rovo-a2a/disconnect").status_code, 403)
            response = client.post("/api/knowledge/jobs", json={})
            self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
