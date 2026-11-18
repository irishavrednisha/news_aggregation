"""Verified TLS escalation for a single public HTTPS URL.

1. The caller tries ordinary requests first.
2. Optional Windows/system roots through truststore, WITHOUT global SSL injection.
3. Optional verified, per-exact-host PEM CA bundle.
Never use requests.verify=False. Browser diagnostic bypass lives in the collector.
"""
from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

# Legacy single NEWS_CA_BUNDLE is restricted to these exact names.
LEGACY_CA_HOSTS = frozenset({
    "gazprom.ru", "www.gazprom.ru",
    "rostec.ru", "www.rostec.ru",
    "rosneft.ru", "www.rosneft.ru", "limited.rosneft.ru",
})


def ca_bundle_for_url(url: str) -> str | None:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or parts.username or parts.password or not host:
        return None
    raw = os.environ.get("NEWS_SOURCE_CA_BUNDLES", "").strip()
    mapping = json.loads(raw) if raw else {}
    if not isinstance(mapping, dict):
        raise ValueError("NEWS_SOURCE_CA_BUNDLES must be a JSON object")
    value = mapping.get(host)
    if not value and host in LEGACY_CA_HOSTS:
        value = os.environ.get("NEWS_CA_BUNDLE", "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"CA bundle for {host} is missing")
    return str(path)


class SystemTrustAdapter(HTTPAdapter):
    def __init__(self, context, **kwargs):
        self.context = context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **kwargs):
        kwargs["ssl_context"] = self.context
        return super().proxy_manager_for(proxy, **kwargs)


def retry_with_trusted_roots(url, headers, timeout, record_issue=None):
    """Return verified response or None. Only called AFTER default TLS error."""
    def report(kind, detail):
        if record_issue is not None:
            record_issue(url, kind, str(detail)[:150])

    if urlsplit(url).scheme != "https":
        return None
    try:
        import truststore
    except ImportError:
        report("system_ca_unavailable", "truststore not installed")
    else:
        try:
            ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            with requests.Session() as http:
                http.mount("https://", SystemTrustAdapter(ctx))
                return http.get(url, headers=headers, timeout=timeout, verify=True)
        except requests.exceptions.RequestException as exc:
            report("system_ca_failed", type(exc).__name__)
        except (ssl.SSLError, OSError) as exc:
            report("system_ca_failed", type(exc).__name__)
    try:
        bundle = ca_bundle_for_url(url)
    except (ValueError, json.JSONDecodeError) as exc:
        report("site_ca_configuration_error", type(exc).__name__)
        return None
    if bundle:
        try:
            return requests.get(url, headers=headers, timeout=timeout, verify=bundle)
        except requests.exceptions.RequestException as exc:
            report("site_ca_failed", type(exc).__name__)
    return None
