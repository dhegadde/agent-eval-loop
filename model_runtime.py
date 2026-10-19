"""Provider-neutral model configuration for EvoAgentX roles.

The executor, optimizer, and judge are configured independently.  The default
configuration is intentionally identical to the original OpenAI-only wiring.
An ``openai_compatible`` role points LiteLLM at a self-hosted endpoint such as
vLLM without granting that endpoint any tools or external authority.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlparse

from evoagentx.models import LiteLLM, LiteLLMConfig


class ModelRuntimeConfigError(ValueError):
    """Raised when a model role cannot be configured safely."""


ROLE_DEFAULTS = {
    "executor": {"model": "gpt-5.4-nano", "max_tokens": 2000},
    "optimizer": {"model": "gpt-5.4", "max_tokens": 8000},
    "judge": {"model": "gpt-5.4", "max_tokens": 1000},
}
SUPPORTED_PROVIDERS = {"openai", "openai_compatible"}
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
REQUEST_PROFILES = {
    "generic": None,
    "qwen_non_thinking": {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
    },
}


@dataclass(frozen=True)
class ModelRoleSettings:
    """Validated settings for one role; secrets are omitted from repr/audits."""

    role: str
    provider: str
    model: str
    max_tokens: int
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    request_profile: str = "generic"

    def audit_dict(self) -> dict[str, object]:
        """Return a serializable, credential-free configuration record."""
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base,
            "authenticated": bool(self.api_key),
            "request_profile": self.request_profile,
        }


@dataclass(frozen=True)
class ModelRoleRuntime:
    settings: ModelRoleSettings
    config: LiteLLMConfig
    llm: LiteLLM


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRuntimeConfigError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _positive_integer(value: object, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeConfigError(f"{field_name} must be a positive integer.") from exc
    if isinstance(value, bool) or parsed <= 0:
        raise ModelRuntimeConfigError(f"{field_name} must be a positive integer.")
    return parsed


def validate_openai_compatible_endpoint(
    api_base: str, api_key: str | None
) -> tuple[str, str | None]:
    """Validate an endpoint without ever probing it.

    Plain HTTP and unauthenticated access are accepted only for loopback.  This
    prevents an accidental cloud deployment from exposing model inference over
    an unencrypted or unauthenticated public endpoint.
    """
    normalized_base = _nonempty(api_base, "api_base").rstrip("/")
    parsed = urlparse(normalized_base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ModelRuntimeConfigError(
            "api_base must be an absolute http(s) URL, normally ending in /v1."
        )
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ModelRuntimeConfigError(
            "api_base must not embed credentials, parameters, a query, or a fragment; "
            "pass authentication through the API-key setting."
        )
    is_loopback = parsed.hostname.lower() in LOOPBACK_HOSTS
    if not is_loopback and parsed.scheme != "https":
        raise ModelRuntimeConfigError(
            "Remote OpenAI-compatible endpoints must use HTTPS."
        )
    normalized_key = api_key.strip() if isinstance(api_key, str) else None
    normalized_key = normalized_key or None
    if not is_loopback and normalized_key is None:
        raise ModelRuntimeConfigError(
            "Remote OpenAI-compatible endpoints require an API key."
        )
    return normalized_base, normalized_key


def endpoint_origin_for_audit(value: str | None) -> str | None:
    """Return a credential- and path-free endpoint origin for logs/contracts."""
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "<configured-invalid-endpoint>"
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "<configured-invalid-endpoint>"
    return f"{parsed.scheme.lower()}://{host}{port}"


def request_extra_body(profile: str) -> dict[str, object] | None:
    """Return a copy of a reviewed, non-secret provider request profile."""
    if profile not in REQUEST_PROFILES:
        raise ModelRuntimeConfigError(
            f"request_profile must be one of {sorted(REQUEST_PROFILES)}."
        )
    value = REQUEST_PROFILES[profile]
    return copy.deepcopy(value) if value is not None else None


def load_role_settings(
    role: str, env: Mapping[str, str] | None = None
) -> ModelRoleSettings:
    """Load one role from environment values without constructing a client."""
    if role not in ROLE_DEFAULTS:
        raise ModelRuntimeConfigError(
            f"Unknown model role {role!r}; expected one of {sorted(ROLE_DEFAULTS)}."
        )
    if env is None:
        import os

        env = os.environ

    prefix = f"EVOAGENTX_{role.upper()}_"
    provider = env.get(f"{prefix}PROVIDER", "openai").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ModelRuntimeConfigError(
            f"{prefix}PROVIDER must be one of {sorted(SUPPORTED_PROVIDERS)}."
        )
    model = _nonempty(
        env.get(f"{prefix}MODEL", ROLE_DEFAULTS[role]["model"]),
        f"{prefix}MODEL",
    )
    max_tokens = _positive_integer(
        env.get(f"{prefix}MAX_TOKENS", str(ROLE_DEFAULTS[role]["max_tokens"])),
        f"{prefix}MAX_TOKENS",
    )
    request_profile = env.get(f"{prefix}REQUEST_PROFILE", "generic").strip()
    request_extra_body(request_profile)

    if provider == "openai":
        if request_profile != "generic":
            raise ModelRuntimeConfigError(
                f"{prefix}REQUEST_PROFILE={request_profile!r} requires "
                "provider=openai_compatible."
            )
        key = env.get(f"{prefix}API_KEY") or env.get("OPENAI_API_KEY")
        key = key.strip() if isinstance(key, str) else ""
        if not key:
            raise ModelRuntimeConfigError(
                f"{prefix}API_KEY or OPENAI_API_KEY is required for provider=openai."
            )
        return ModelRoleSettings(
            role=role,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            api_key=key,
            request_profile=request_profile,
        )

    api_base = env.get(f"{prefix}API_BASE") or env.get(
        "EVOAGENTX_OPENAI_COMPATIBLE_API_BASE"
    )
    api_key = env.get(f"{prefix}API_KEY") or env.get(
        "EVOAGENTX_OPENAI_COMPATIBLE_API_KEY"
    )
    if not api_base:
        raise ModelRuntimeConfigError(
            f"{prefix}API_BASE or EVOAGENTX_OPENAI_COMPATIBLE_API_BASE is required."
        )
    api_base, api_key = validate_openai_compatible_endpoint(api_base, api_key)
    return ModelRoleSettings(
        role=role,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key,
        request_profile=request_profile,
    )


def build_litellm_config(settings: ModelRoleSettings) -> LiteLLMConfig:
    """Translate validated settings into EvoAgentX's installed config shape."""
    common = {
        "max_tokens": settings.max_tokens,
        "stream": False,
    }
    extra_body = request_extra_body(settings.request_profile)
    if extra_body is not None:
        common["extra_body"] = extra_body
    if settings.provider == "openai":
        return LiteLLMConfig(
            model=settings.model,
            openai_key=settings.api_key,
            **common,
        )

    # LiteLLM's ``openai/`` prefix selects its generic OpenAI-compatible route;
    # EvoAgentX's is_local flag suppresses provider-specific key requirements and
    # records token usage with zero fabricated per-token cost.
    routed_model = settings.model
    if not routed_model.startswith("openai/"):
        routed_model = f"openai/{routed_model}"
    return LiteLLMConfig(
        model=routed_model,
        api_base=settings.api_base,
        api_key=settings.api_key or "EMPTY",
        is_local=True,
        **common,
    )


def build_role_runtime(
    role: str, env: Mapping[str, str] | None = None
) -> ModelRoleRuntime:
    settings = load_role_settings(role, env)
    config = build_litellm_config(settings)
    return ModelRoleRuntime(
        settings=settings,
        config=config,
        llm=LiteLLM(config=config),
    )


def runtime_manifest(*settings: ModelRoleSettings) -> dict[str, object]:
    """Credential-free snapshot suitable for logs and benchmark artifacts."""
    return {
        "schema_version": 1,
        "roles": {item.role: item.audit_dict() for item in settings},
    }


__all__ = [
    "ModelRoleRuntime",
    "ModelRoleSettings",
    "ModelRuntimeConfigError",
    "ROLE_DEFAULTS",
    "REQUEST_PROFILES",
    "build_litellm_config",
    "build_role_runtime",
    "endpoint_origin_for_audit",
    "load_role_settings",
    "runtime_manifest",
    "request_extra_body",
    "validate_openai_compatible_endpoint",
]
