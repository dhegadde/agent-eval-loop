"""Zero-network tests for provider-neutral model configuration."""

from __future__ import annotations

from model_runtime import (
    ModelRuntimeConfigError,
    build_litellm_config,
    endpoint_origin_for_audit,
    load_role_settings,
    runtime_manifest,
    request_extra_body,
    validate_openai_compatible_endpoint,
)


def expect_error(fn, text: str) -> None:
    try:
        fn()
    except ModelRuntimeConfigError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected ModelRuntimeConfigError containing {text!r}")


def main() -> None:
    defaults = load_role_settings("executor", {"OPENAI_API_KEY": "secret-default"})
    assert defaults.provider == "openai"
    assert defaults.model == "gpt-5.4-nano"
    assert defaults.max_tokens == 2000
    default_config = build_litellm_config(defaults)
    assert default_config.model == "gpt-5.4-nano"
    assert default_config.openai_key == "secret-default"
    assert default_config.is_local is False

    local_env = {
        "EVOAGENTX_EXECUTOR_PROVIDER": "openai_compatible",
        "EVOAGENTX_EXECUTOR_MODEL": "Qwen/Qwen3.8-27B-FP8",
        "EVOAGENTX_EXECUTOR_API_BASE": "https://gpu.example.test/v1/",
        "EVOAGENTX_EXECUTOR_API_KEY": "private-endpoint-key",
        "EVOAGENTX_EXECUTOR_MAX_TOKENS": "4096",
        "EVOAGENTX_EXECUTOR_REQUEST_PROFILE": "qwen_non_thinking",
    }
    local = load_role_settings("executor", local_env)
    assert local.api_base == "https://gpu.example.test/v1"
    assert local.max_tokens == 4096
    assert local.request_profile == "qwen_non_thinking"
    local_config = build_litellm_config(local)
    assert local_config.model == (
        "openai/Qwen/Qwen3.8-27B-FP8"
    )
    assert local_config.api_base == "https://gpu.example.test/v1"
    assert local_config.api_key == "private-endpoint-key"
    assert local_config.is_local is True
    assert local_config.get_set_params()["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
    }
    profile_copy = request_extra_body("qwen_non_thinking")
    profile_copy["chat_template_kwargs"]["enable_thinking"] = True
    assert request_extra_body("qwen_non_thinking")["chat_template_kwargs"][
        "enable_thinking"
    ] is False

    manifest = runtime_manifest(defaults, local)
    encoded = repr(manifest)
    assert "secret-default" not in encoded
    assert "private-endpoint-key" not in encoded
    assert manifest["roles"]["executor"]["authenticated"] is True

    base, key = validate_openai_compatible_endpoint(
        "http://127.0.0.1:8000/v1", None
    )
    assert base == "http://127.0.0.1:8000/v1"
    assert key is None
    assert endpoint_origin_for_audit(
        "https://user:secret@gpu.example.test:443/v1?token=secret"
    ) == "https://gpu.example.test:443"

    expect_error(
        lambda: request_extra_body("invented-profile"),
        "must be one of",
    )
    expect_error(
        lambda: load_role_settings(
            "judge",
            {
                "OPENAI_API_KEY": "x",
                "EVOAGENTX_JUDGE_REQUEST_PROFILE": "qwen_non_thinking",
            },
        ),
        "requires provider=openai_compatible",
    )
    expect_error(
        lambda: load_role_settings(
            "judge", {"EVOAGENTX_JUDGE_PROVIDER": "openai_compatible"}
        ),
        "API_BASE",
    )
    expect_error(
        lambda: validate_openai_compatible_endpoint(
            "http://gpu.example.test/v1", "key"
        ),
        "HTTPS",
    )
    expect_error(
        lambda: validate_openai_compatible_endpoint(
            "https://gpu.example.test/v1", None
        ),
        "require an API key",
    )
    expect_error(
        lambda: validate_openai_compatible_endpoint(
            "https://user:secret@gpu.example.test/v1", "key"
        ),
        "must not embed credentials",
    )
    expect_error(
        lambda: validate_openai_compatible_endpoint(
            "https://gpu.example.test/v1?token=secret", "key"
        ),
        "must not embed credentials",
    )
    expect_error(
        lambda: load_role_settings(
            "executor",
            {
                "OPENAI_API_KEY": "x",
                "EVOAGENTX_EXECUTOR_MAX_TOKENS": "0",
            },
        ),
        "positive integer",
    )

    print("ALL MODEL-RUNTIME TESTS PASSED (zero network calls).")


if __name__ == "__main__":
    main()
