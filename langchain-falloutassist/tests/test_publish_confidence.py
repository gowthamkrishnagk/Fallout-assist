"""Confidence-based Jira publication behavior."""

import unittest
from unittest.mock import patch

from falloutassist_next import workflow


class PublishConfidenceTests(unittest.TestCase):
    def test_medium_confidence_comment_contains_action_and_review_caveat(self):
        step = {
            "text": "Resolve the failed create order, then resend the ChangeMsisdn order.",
            "source_key": "SAC-260275",
            "source_excerpt": "Resolve the create order, then send the ChangeMsisdn order.",
        }
        state = {
            "settings": {"public_base_url": "https://falloutassist.example"},
            "issue": {"key": "SAC-260434"},
            "rule": None,
            "tier": "medium",
            "draft": {"decline": False, "steps": [step]},
            "supervisor_result": {
                "confidence": "medium", "draft": {"decline": False, "steps": [step]},
            },
            "candidates": [{
                "key": "SAC-260275", "url": "https://jira.example/browse/SAC-260275",
                "composite_score": 0.875,
            }],
            "history": {},
        }

        result = workflow._comment(state)

        self.assertIn("Suggested workaround", result["comment_body"])
        self.assertIn(step["text"], result["comment_body"])
        self.assertIn("medium confidence", result["comment_body"])
        self.assertIn("Verify the order", result["comment_body"])
        self.assertNotIn("manual verification needed", result["comment_body"])

    def test_low_confidence_stays_review_only_even_when_publishing_is_enabled(self):
        state = {
            "settings": {"publish_enabled": True, "dry_run": False},
            "publish_requested": True, "tier": "low", "id": "review-id",
            "issue_key": "SAC-260434", "actor": "tester",
        }
        with patch.object(workflow.storage, "update_investigation_status") as update, \
             patch.object(workflow.storage, "audit") as audit, \
             patch.object(workflow, "_jira") as jira:
            result = workflow._publish(state)

        self.assertEqual(result, {"status": "preview"})
        update.assert_called_once_with("review-id", "preview")
        audit.assert_called_once()
        jira.assert_not_called()

    def test_medium_confidence_publishes_when_scan_publishing_is_enabled(self):
        state = {
            "settings": {"publish_enabled": True, "dry_run": False,
                         "public_base_url": "https://falloutassist.example"},
            "publish_requested": True, "tier": "medium", "id": "publish-id",
            "issue_key": "SAC-260434", "actor": "tester",
            "issue": {"key": "SAC-260434", "updated": "2026-10-20T12:00:00Z"},
            "comment_body": "Suggested workaround — medium confidence",
        }
        with patch.object(workflow, "feedback_secret", return_value="test-secret"), \
             patch.object(workflow.storage, "publication", return_value=None), \
             patch.object(workflow.storage, "save_publication") as save_publication, \
             patch.object(workflow.storage, "update_investigation_status") as update, \
             patch.object(workflow.storage, "audit"), \
             patch.object(workflow, "_jira") as jira_factory:
            jira = jira_factory.return_value
            jira.existing_comment.return_value = None
            jira.publish_comment.return_value = "comment-123"

            result = workflow._publish(state)

        self.assertEqual(result, {"status": "published", "comment_id": "comment-123"})
        jira.publish_comment.assert_called_once_with("SAC-260434", state["comment_body"], "")
        save_publication.assert_called_once()
        update.assert_called_once_with("publish-id", "published")


if __name__ == "__main__":
    unittest.main()
