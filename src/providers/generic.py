from __future__ import annotations
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional
from ..exceptions import ProviderError, ProviderTimeoutError
from ..policies import Policy
from ..types import GuardDecision, ToolCall
from .base import BaseProvider, ChatMessage, ProviderConfig

_logger = logging.getLogger(__name__)

#: A callable that takes (messages, **kwargs) and returns any response object.
CallFn = Callable[[List[ChatMessage], Any], Any]

#: A callable that extracts a str from a provider response object.
ExtractFn = Callable[[Any], str]

#: Optionally extract tool calls from a response.
ExtractToolsFn = Callable[[Any], List[ToolCall]]

class GenericProvider(BaseProvider):
    provider_name = "generic"
    def __init__(
        self,
        call_fn:            CallFn,
        extract_fn:         ExtractFn,
        extract_tools_fn:   Optional[ExtractToolsFn]  = None,
        provider_name:      str                        = "generic",
        policy:             Optional[Policy]           = None,
        config:             Optional[ProviderConfig]   = None,
    ) -> None:
        # Allow per-instance provider_name without touching the class attribute
        self.provider_name = provider_name  # type: ignore[assignment]

        super().__init__(policy=policy, config=config)

        if not callable(call_fn):
            raise TypeError(
                f"call_fn must be callable, got {type(call_fn).__name__!r}"
            )
        if not callable(extract_fn):
            raise TypeError(
                f"extract_fn must be callable, got {type(extract_fn).__name__!r}"
            )
        if extract_tools_fn is not None and not callable(extract_tools_fn):
            raise TypeError(
                f"extract_tools_fn must be callable or None, "
                f"got {type(extract_tools_fn).__name__!r}"
            )

        self._call_fn          = call_fn
        self._extract_fn       = extract_fn
        self._extract_tools_fn = extract_tools_fn

    def _call_model(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        try:
            return self._call_fn(messages, **kwargs)
        except (ProviderError, ProviderTimeoutError):
            raise  # already wrapped — let propagate
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                provider_name=self.provider_name,
                timeout_seconds=self._config.timeout_seconds,
                original_error=exc,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                message=str(exc),
                provider_name=self.provider_name,
                original_error=exc,
            ) from exc

    def _extract_text(self, response: Any) -> str:
        try:
            result = self._extract_fn(response)
            return result if isinstance(result, str) else str(result) if result else ""
        except Exception as exc:
            _logger.warning(
                "%s: extract_fn raised an exception — returning empty string. "
                "Error: %s", self.provider_name, exc,
            )
            return ""

    def _extract_tool_calls(self, response: Any) -> List[ToolCall]:
        if self._extract_tools_fn is None:
            return []
        try:
            result = self._extract_tools_fn(response)
            return result if isinstance(result, list) else []
        except Exception as exc:
            _logger.warning(
                "%s: extract_tools_fn raised an exception — skipping tool "
                "validation. Error: %s", self.provider_name, exc,
            )
            return []

    @classmethod
    def from_openai_compatible_url(
        cls,
        url:           str,
        model:         str,
        api_key:       str                       = "",
        provider_name: str                       = "generic-openai-compat",
        policy:        Optional[Policy]          = None,
        config:        Optional[ProviderConfig]  = None,
        extra_headers: Optional[Dict[str, str]]  = None,
    ) -> "GenericProvider":
        resolved_config = config or ProviderConfig()
        timeout         = resolved_config.timeout_seconds

        # Build headers once at construction time
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)

        def _call(messages: List[ChatMessage], **kwargs: Any) -> Dict[str, Any]:
            body: Dict[str, Any] = {
                "model":    kwargs.pop("model", model),
                "messages": messages,
                **kwargs,
            }
            payload = json.dumps(body).encode("utf-8")
            req     = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body_text = ""
                try:
                    body_text = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                raise ProviderError(
                    message=(
                        f"HTTP {exc.code} from {provider_name}: "
                        f"{exc.reason}. Body: {body_text}"
                    ),
                    provider_name=provider_name,
                    status_code=exc.code,
                    original_error=exc,
                ) from exc
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    provider_name=provider_name,
                    timeout_seconds=timeout,
                    original_error=exc,
                ) from exc
            except Exception as exc:
                raise ProviderError(
                    message=f"Request to {provider_name} failed: {exc}",
                    provider_name=provider_name,
                    original_error=exc,
                ) from exc

        def _extract(response: Dict[str, Any]) -> str:
            try:
                return response["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                return ""

        return cls(
            call_fn=_call,
            extract_fn=_extract,
            provider_name=provider_name,
            policy=policy,
            config=resolved_config,
        )

    @classmethod
    def from_callable(
        cls,
        fn:            Callable[..., str],
        provider_name: str                      = "callable",
        policy:        Optional[Policy]         = None,
        config:        Optional[ProviderConfig] = None,
    ) -> "GenericProvider":
        return cls(
            call_fn=fn,
            extract_fn=lambda r: r if isinstance(r, str) else str(r),
            provider_name=provider_name,
            policy=policy,
            config=config,
        )

class CallableMixin:
    def _call_model(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        call_fn = getattr(self, "_call_fn", None)
        if not callable(call_fn):
            raise ProviderError(
                message=(
                    f"{self.__class__.__name__}: _call_fn is not set or not callable. "
                    "Set self._call_fn before calling chat()."
                ),
                provider_name=getattr(self, "provider_name", "unknown"),
            )
        try:
            return call_fn(messages, **kwargs)
        except (ProviderError, ProviderTimeoutError):
            raise
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                provider_name=getattr(self, "provider_name", "unknown"),
                timeout_seconds=getattr(
                    getattr(self, "_config", None), "timeout_seconds", 30.0
                ),
                original_error=exc,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                message=str(exc),
                provider_name=getattr(self, "provider_name", "unknown"),
                original_error=exc,
            ) from exc

    def _extract_text(self, response: Any) -> str:
        extract_fn = getattr(self, "_extract_fn", None)
        if not callable(extract_fn):
            _logger.warning(
                "%s: _extract_fn not set — returning empty string",
                self.__class__.__name__,
            )
            return ""
        try:
            result = extract_fn(response)
            return result if isinstance(result, str) else str(result) if result else ""
        except Exception as exc:
            _logger.warning(
                "%s: _extract_fn raised %s — returning empty string",
                self.__class__.__name__, exc,
            )
            return ""

    def _extract_tool_calls(self, response: Any) -> List[ToolCall]:
        extract_tools_fn = getattr(self, "_extract_tools_fn", None)
        if not callable(extract_tools_fn):
            return []
        try:
            result = extract_tools_fn(response)
            return result if isinstance(result, list) else []
        except Exception as exc:
            _logger.warning(
                "%s: _extract_tools_fn raised %s — skipping tool validation",
                self.__class__.__name__, exc,
            )
            return []

__all__ = [
    "GenericProvider",
    "CallableMixin",
]