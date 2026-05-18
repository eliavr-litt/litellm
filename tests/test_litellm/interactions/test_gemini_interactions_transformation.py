"""
Tests for Gemini Interactions API transformation.

Covers:
- validate_environment: x-goog-api-key header, Api-Revision schema selection
- get_complete_url: API key excluded from URL
- get/delete/cancel interaction request URLs
- transform_request: response_mime_type coalescing, image_config migration
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
)
from litellm.types.router import GenericLiteLLMParams

_PATCH_GET_API_KEY = "litellm.llms.gemini.common_utils.GeminiModelInfo.get_api_key"


@pytest.fixture
def config():
    return GoogleAIStudioInteractionsConfig()


class TestValidateEnvironment:
    def test_sets_x_goog_api_key_header(self, config):
        litellm_params = GenericLiteLLMParams(api_key="test-api-key-123")

        headers = config.validate_environment(
            headers={},
            model="gemini-2.5-flash",
            litellm_params=litellm_params,
        )

        assert headers["x-goog-api-key"] == "test-api-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_skips_header(self, config):
        litellm_params = GenericLiteLLMParams(api_key=None)

        with patch(_PATCH_GET_API_KEY, return_value=None):
            headers = config.validate_environment(
                headers={},
                model="gemini-2.5-flash",
                litellm_params=litellm_params,
            )

        assert "x-goog-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_no_litellm_params_skips_header(self, config):
        headers = config.validate_environment(
            headers={},
            model="gemini-2.5-flash",
            litellm_params=None,
        )

        assert "x-goog-api-key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_preserves_existing_headers(self, config):
        litellm_params = GenericLiteLLMParams(api_key="test-key")

        headers = config.validate_environment(
            headers={"X-Custom": "value"},
            model="gemini-2.5-flash",
            litellm_params=litellm_params,
        )

        assert headers["X-Custom"] == "value"
        assert headers["x-goog-api-key"] == "test-key"

    def test_api_revision_new_schema_by_default(self, config):
        # Default: use_legacy_interactions_schema=False → new steps schema
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            headers = config.validate_environment(
                headers={}, model="gemini-2.5-flash", litellm_params=None
            )
            assert headers["Api-Revision"] == "2026-05-20"
        finally:
            litellm.use_legacy_interactions_schema = original

    def test_api_revision_legacy_schema_when_flag_set(self, config):
        # Flag on → legacy outputs schema until June 8, 2026
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = True
            headers = config.validate_environment(
                headers={}, model="gemini-2.5-flash", litellm_params=None
            )
            assert headers["Api-Revision"] == "2026-05-07"
        finally:
            litellm.use_legacy_interactions_schema = original


class TestGetCompleteUrl:
    def test_url_excludes_api_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url = config.get_complete_url(
                api_base=None,
                model="gemini-2.5-flash",
                litellm_params={"api_key": "secret-key"},
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert url.endswith("/interactions")

    def test_stream_url_has_alt_sse_only(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url = config.get_complete_url(
                api_base=None,
                model="gemini-2.5-flash",
                litellm_params={"api_key": "secret-key"},
                stream=True,
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert "alt=sse" in url

    def test_raises_without_api_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value=None):
            with pytest.raises(ValueError, match="Google API key is required"):
                config.get_complete_url(
                    api_base=None,
                    model="gemini-2.5-flash",
                    litellm_params={"api_key": None},
                )


class TestInteractionOperationUrls:
    """Test that get/delete/cancel interaction URLs exclude API key."""

    @pytest.mark.parametrize(
        "method_name,interaction_id,expected_suffix",
        [
            ("transform_get_interaction_request", "interaction-123", "interaction-123"),
            (
                "transform_delete_interaction_request",
                "interaction-456",
                "interaction-456",
            ),
            (
                "transform_cancel_interaction_request",
                "interaction-789",
                "interaction-789:cancel",
            ),
        ],
    )
    def test_url_excludes_key(
        self, config, method_name, interaction_id, expected_suffix
    ):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url, params = getattr(config, method_name)(
                interaction_id=interaction_id,
                api_base="https://generativelanguage.googleapis.com",
                litellm_params=GenericLiteLLMParams(api_key="secret-key"),
                headers={},
            )

        assert "key=" not in url
        assert "secret-key" not in url
        assert expected_suffix in url

    def test_interaction_id_is_encoded_as_one_path_segment(self, config):
        with patch(_PATCH_GET_API_KEY, return_value="secret-key"):
            url, params = config.transform_cancel_interaction_request(
                interaction_id="../../interactions/other?x=1#frag",
                api_base="https://generativelanguage.googleapis.com",
                litellm_params=GenericLiteLLMParams(api_key="secret-key"),
                headers={},
            )

        assert (
            url
            == "https://generativelanguage.googleapis.com/v1beta/interactions/..%2F..%2Finteractions%2Fother%3Fx%3D1%23frag:cancel"
        )
        assert params == {}

    def test_get_interaction_raises_without_key(self, config):
        with patch(_PATCH_GET_API_KEY, return_value=None):
            with pytest.raises(ValueError, match="Google API key is required"):
                config.transform_get_interaction_request(
                    interaction_id="interaction-123",
                    api_base="https://generativelanguage.googleapis.com",
                    litellm_params=GenericLiteLLMParams(api_key=None),
                    headers={},
                )


class TestTransformRequestSchemaCoalescing:
    """Test new-schema request coalescing (Api-Revision: 2026-05-20)."""

    def test_response_mime_type_folded_into_response_format(self, config):
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="summarise",
                optional_params={
                    "response_mime_type": "application/json",
                    "response_format": {"type": "object", "properties": {}},
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        # response_mime_type must not appear as a top-level body key
        assert "response_mime_type" not in body
        rf = body["response_format"]
        assert rf["type"] == "text"
        assert rf["mime_type"] == "application/json"
        assert "schema" in rf

    def test_image_config_moved_to_response_format(self, config):
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = False
            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="draw a sunset",
                optional_params={
                    "generation_config": {
                        "temperature": 0.7,
                        "image_config": {"aspect_ratio": "1:1", "image_size": "1K"},
                    }
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        # image_config removed from generation_config
        assert "image_config" not in body.get("generation_config", {})
        # moved into response_format with type=image
        rf = body["response_format"]
        assert rf["type"] == "image"
        assert rf["aspect_ratio"] == "1:1"

    def test_legacy_schema_passes_fields_unchanged(self, config):
        original = litellm.use_legacy_interactions_schema
        try:
            litellm.use_legacy_interactions_schema = True
            body = config.transform_request(
                model="gemini/gemini-2.5-flash",
                agent=None,
                input="hello",
                optional_params={
                    "response_mime_type": "application/json",
                    "generation_config": {"image_config": {"aspect_ratio": "16:9"}},
                },
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        finally:
            litellm.use_legacy_interactions_schema = original

        assert body["response_mime_type"] == "application/json"
        assert body["generation_config"]["image_config"]["aspect_ratio"] == "16:9"
