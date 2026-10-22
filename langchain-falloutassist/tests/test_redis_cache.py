"""RedisVL contract tests. Real query/schema code, mocked Redis transport and LLM."""

import copy
import json
import time
from unittest.mock import MagicMock, patch

from redis import Redis
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import CustomVectorizer

from falloutassist_next import redis_cache, semantic_cache, storage
from test_semantic_cache import CacheTestCase, QUERY, CANDIDATES


class RedisCacheTests(CacheTestCase):
    def setUp(self):
        super().setUp()
        self.settings["semantic_cache_backend"] = "redis"
        self.client = Redis.from_url("redis://localhost:6379", socket_timeout=1)
        self.addCleanup(self.client.close)
        self.transport = MagicMock()
        # Keep RedisVL's actual schema, validation, store and query generation.
        self.cache = SemanticCache(
            name="fa_rerank_{test}_model_v1", redis_client=self.client,
            vectorizer=CustomVectorizer(embed=lambda _: [1.0, 0.0]),
            filterable_fields=[{"name": "scope", "type": "tag"},
                               {"name": "query_hash", "type": "tag"},
                               {"name": "mode", "type": "tag"}], create_index=False)
        self.backend = redis_cache.RedisRankingCache(self.cache, self.transport, "test", 2)
        self.query = patch.object(self.cache.index, "query", return_value=[]).start()
        self.load = patch.object(self.cache.index, "load", return_value=[
            "fa_rerank_{test}_model_v1:entry"]).start()
        self.factory = redis_cache._cache
        patch.object(redis_cache, "_cache", return_value=self.backend).start()
        patch.object(redis_cache, "config_value", side_effect={
            "SEMANTIC_CACHE_REDIS_URL": "redis://localhost:6379",
            "SEMANTIC_CACHE_NAMESPACE": "test"}.__getitem__).start()

    def stored_hit(self, *, semantic=False, **overrides):
        saved = self.load.call_args.kwargs["data"][0]
        hit = {key: value for key, value in saved.items() if key != "prompt_vector"}
        hit["metadata"] = json.loads(hit["metadata"])
        if semantic:
            hit.update(id="fa_rerank_{test}_model_v1:entry", vector_distance=0.01)
        hit.update(overrides)
        return hit

    def test_real_redisvl_factory_uses_existing_embeddings_and_shared_namespace(self):
        def construct(**kwargs):
            kwargs["create_index"] = False
            return SemanticCache(**kwargs)
        self.factory.cache_clear()
        self.addCleanup(self.factory.cache_clear)
        with patch.object(redis_cache, "_client", return_value=self.client), \
                patch.object(self.client, "ping", return_value=True), \
                patch("redisvl.extensions.cache.llm.SemanticCache", side_effect=construct):
            first = self.factory("redis://localhost:6379", "test", "test-model")
            second = self.factory("redis://localhost:6379", "test", "test-model")
        self.assertIs(first, second)
        self.assertEqual(first.dims, 2)
        self.assertEqual(first.cache.ttl, None)  # Reads must not refresh the absolute TTL.
        self.assertIn("{test}", first.cache.index.schema.index.prefix)
        self.embed.assert_called_once_with("dimension test", "test-model")

    def test_exact_hit_avoids_query_embedding_and_model(self):
        first, _ = self.rank()
        self.query.return_value = [self.stored_hit()]
        self.embed.reset_mock()
        second, info = self.rank()
        self.assertEqual(first, second)
        self.assertEqual(info["backend"], "redis")
        self.assertEqual(info["status"], "exact_hit")
        self.embed.assert_not_called()
        self.assertEqual(self.invoke.call_count, 1)

    def test_semantic_hit_uses_vector_distance_and_partition_filter(self):
        self.rank()
        hit = self.stored_hit(semantic=True)
        self.query.side_effect = [[], [hit]]
        with patch.object(self.cache, "check", wraps=self.cache.check) as check:
            _, info = self.rank(QUERY.replace("Customer", "Client"))
        self.assertEqual(info["status"], "semantic_hit")
        self.assertAlmostEqual(info["similarity"], 0.99)
        self.assertAlmostEqual(check.call_args.kwargs["distance_threshold"], 0.03)
        expression = str(check.call_args.kwargs["filter_expression"])
        self.assertIn("@scope:", expression)
        self.assertIn("@mode:{semantic}", expression)
        self.assertIn("@inserted_at:", expression)
        self.assertEqual(self.invoke.call_count, 1)

    def test_redis_writes_hashes_ttl_mode_and_capacity(self):
        _, info = self.rank()
        data = self.load.call_args.kwargs["data"][0]
        self.assertEqual(info["status"], "miss")
        self.assertEqual(self.load.call_args.kwargs["ttl"], 3600)
        self.assertEqual(len(data["prompt"]), 64)
        self.assertNotIn("Customer provisioning", json.dumps(data, default=str))
        self.assertEqual(data["mode"], "semantic")
        self.transport.eval.assert_called_once()
        self.assertEqual(self.transport.eval.call_args.args[-1], 1000)
        self.assertIn("{test}", self.transport.eval.call_args.args[2])

    def test_incomplete_query_or_embedding_failure_only_stores_exact_entries(self):
        self.rank(QUERY.replace("Order Type: Change\n", ""))
        self.assertEqual(self.load.call_args.kwargs["data"][0]["mode"], "exact")
        self.embed.assert_not_called()
        self.embed.side_effect = RuntimeError("embedding failed")
        _, info = self.rank()
        self.assertEqual(info["embedding_error"], "RuntimeError")
        self.assertEqual(self.load.call_args.kwargs["data"][0]["mode"], "exact")

    def test_wrong_scope_stale_or_invalid_score_is_never_reused(self):
        self.rank()
        original = self.stored_hit()
        mutations = [lambda row: row["metadata"].update(scope="other-tenant"),
                     lambda row: row["metadata"].update(created_at=time.time()-4000),
                     lambda row: row.update(response='{"foreign":{}}')]
        for mutate in mutations:
            hit = copy.deepcopy(original)
            mutate(hit)
            self.query.side_effect = [[hit], []]
            self.assertEqual(self.rank()[1]["status"], "miss")
        self.assertEqual(self.invoke.call_count, 4)

    def test_redis_outage_and_write_failure_do_not_block_model(self):
        self.query.side_effect = RuntimeError("Redis unavailable")
        self.assertEqual(self.rank()[1]["status"], "unavailable")
        self.query.side_effect = None
        self.query.return_value = []
        self.transport.eval.side_effect = RuntimeError("capacity update failed")
        _, info = self.rank()
        self.assertEqual(info["write_error"], "RuntimeError")
        self.transport.unlink.assert_called_once_with("fa_rerank_{test}_model_v1:entry")
        self.assertEqual(self.invoke.call_count, 2)

    def test_clear_invalidates_inflight_writes_and_cleans_only_namespace(self):
        scopes = []
        def record(scope, *args):
            scopes.append(scope)
            return []
        with patch.object(self.backend, "exact", side_effect=record):
            self.rank()
            storage.clear_semantic_cache("tester")
            self.rank()
        self.assertNotEqual(scopes[0], scopes[1])
        self.transport.scan_iter.return_value = ["fa_rerank_{test}_old:one"]
        with patch.object(redis_cache, "_client", return_value=self.transport):
            redis_cache.clear_remote()
        self.transport.scan_iter.assert_called_once_with(match="fa_rerank_{test}_*:*", count=200)
        self.transport.unlink.assert_any_call("fa_rerank_{test}_old:one")

    def test_redis_connection_url_is_encrypted_and_not_returned_to_browser(self):
        from unittest.mock import patch as env_patch
        import os
        with env_patch.dict(os.environ, {"SESSION_SECRET": "isolated-cache-test"}):
            secret = "rediss://default:private-password@example.test:6379/0"
            snapshot = storage.update_connection_config({"SEMANTIC_CACHE_REDIS_URL": secret}, "tester")
            self.assertNotIn("SEMANTIC_CACHE_REDIS_URL", snapshot["values"])
            self.assertTrue(snapshot["secrets"]["SEMANTIC_CACHE_REDIS_URL"])
            with storage.connect() as db:
                saved = db.execute("SELECT value FROM connection_config WHERE key=?", (
                    "SEMANTIC_CACHE_REDIS_URL",)).fetchone()["value"]
            self.assertNotIn("private-password", saved)
            self.assertEqual(storage.connection_value("SEMANTIC_CACHE_REDIS_URL"), secret)
        for changes in ({"SEMANTIC_CACHE_REDIS_URL": "https://example.test"},
                        {"SEMANTIC_CACHE_REDIS_URL": "rediss://example.test?ssl_cert_reqs=none"},
                        {"SEMANTIC_CACHE_NAMESPACE": "*"}):
            with self.assertRaises(ValueError):
                storage.update_connection_config(changes, "tester")

    def test_clear_endpoint_invalidates_even_when_redis_is_unavailable(self):
        import os
        with patch.dict(os.environ, {"SESSION_SECRET": "isolated-test-session-key"}):
            from fastapi.testclient import TestClient
            from falloutassist_next import web
        storage.update_settings({"semantic_cache_backend": "redis"}, "tester")
        web.app.dependency_overrides[web._admin] = lambda: {"email": "tester"}
        try:
            with TestClient(web.app) as client:
                token = client.get("/api/session").json()["csrf"]
                with patch.object(redis_cache, "clear_remote", side_effect=RuntimeError("unavailable")):
                    result = client.post("/api/semantic-cache/clear", headers={"x-csrf-token": token})
                self.assertEqual(result.status_code, 200)
                self.assertTrue(result.json()["ok"])
                self.assertIn("invalidated", result.json()["warning"])
                self.assertTrue(storage.get_state("semantic_cache_generation"))
                self.assertEqual(storage.recent_audit(1)[0]["action"], "semantic_cache_remote_clear_failed")
        finally:
            web.app.dependency_overrides.clear()
