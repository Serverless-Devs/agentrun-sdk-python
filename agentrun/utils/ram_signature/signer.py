"""
AgentRun RAM 签名 - Python 手写实现（无外部 signer 依赖）

实现 AGENTRUN4-HMAC-SHA256，与 http-auth-acs / ram-e2e-test（alibabacloud_signer_inner）行为一致：
- UNSIGNED-PAYLOAD
- x-acs-date（ISO8601）、x-acs-content-sha256、host、可选 x-acs-security-token
- Canonical Request 与 http-auth-acs 一致（URI 不编码、Query 空值用 key=、仅签 x-acs-* / host / content-type）
- StringToSign 为 2 行：ALGORITHM + "\\n" + SHA256(CanonicalRequest)（无 timestamp/scope）
"""

from datetime import datetime, timezone
import hashlib
import hmac
from typing import Optional
from urllib.parse import quote, urlparse

ALGORITHM = "AGENTRUN4-HMAC-SHA256"
UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"
SCOPE_SUFFIX = "aliyun_v4_request"
KEY_PREFIX = "aliyun_v4"


def _build_scope(date: str, region: str, product: str) -> str:
    return f"{date}/{region}/{product}/{SCOPE_SUFFIX}"


def _percent_encode(value: Optional[str]) -> str:
    """与 http-auth-acs 一致：quote(safe='') 后把 %7E 还原为 ~。"""
    if value is None:
        return ""
    return quote(str(value), safe="").replace("%7E", "~")


def _canonical_uri(path: str) -> str:
    """与 http-auth-acs 一致：path 原样，不 percent-encode。"""
    if path is None or path == "":
        return "/"
    return path


def _canonical_query(query_params: dict) -> str:
    """与 http-auth-acs 一致：空值为 encoded_key=（带等号）。"""
    if not query_params:
        return ""
    parts = []
    for k in sorted(query_params.keys()):
        v = query_params.get(k)
        enc_k = _percent_encode(k)
        if v is not None and v != "":
            parts.append(f"{enc_k}={_percent_encode(v)}")
        else:
            parts.append(f"{enc_k}=")
    return "&".join(parts)


def _get_signed_headers(headers: dict) -> list[str]:
    """与 http-auth-acs 一致：仅签 x-acs-*、host、content-type，且 value 非 None。"""
    out = set()
    for key, value in headers.items():
        lower_key = key.lower().strip()
        if value is not None and (
            lower_key.startswith("x-acs-")
            or lower_key == "host"
            or lower_key == "content-type"
        ):
            out.add(lower_key)
    return sorted(out)


def _canonical_headers(headers: dict) -> tuple[str, str]:
    """与 http-auth-acs 一致：先归一化再按 signed_headers 顺序输出 header:value\\n。"""
    new_headers: dict[str, str] = {}
    for k, v in headers.items():
        lower_key = k.lower().strip()
        if v is not None:
            new_headers[lower_key] = str(v).strip()
    signed_list = _get_signed_headers(headers)
    canonical = "".join(f"{h}:{new_headers[h]}\n" for h in signed_list)
    signed_str = ";".join(signed_list)
    return canonical, signed_str


def _calc_canonical_request(
    method: str,
    pathname: str,
    query_params: dict,
    headers: dict,
    hashed_payload: str,
) -> str:
    method = method.upper()
    uri = _canonical_uri(pathname)
    query = _canonical_query(query_params)
    canon_headers, signed_headers = _canonical_headers(headers)
    return f"{method}\n{uri}\n{query}\n{canon_headers}\n{signed_headers}\n{hashed_payload}"


def _calc_string_to_sign(canonical_request: str) -> str:
    """与 http-auth-acs 一致：2 行 StringToSign = ALGORITHM + \\n + SHA256(CanonicalRequest)。"""
    digest = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    return f"{ALGORITHM}\n{digest}"


def _get_signing_key(
    secret: str, date: str, region: str, product: str
) -> bytes:
    key = (KEY_PREFIX + secret).encode("utf-8")
    k_date = hmac.new(key, date.encode("utf-8"), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_product = hmac.new(
        k_region, product.encode("utf-8"), hashlib.sha256
    ).digest()
    k_signing = hmac.new(
        k_product, SCOPE_SUFFIX.encode("utf-8"), hashlib.sha256
    ).digest()
    return k_signing


def _calc_signature(
    secret: str,
    date: str,
    region: str,
    product: str,
    string_to_sign: str,
) -> str:
    signing_key = _get_signing_key(secret, date, region, product)
    return hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def get_agentrun_signed_headers(
    url: str,
    method: str = "GET",
    access_key_id: Optional[str] = None,
    access_key_secret: Optional[str] = None,
    security_token: Optional[str] = None,
    region: str = "cn-hangzhou",
    product: str = "agentrun",
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
    sign_time: Optional[datetime] = None,
) -> dict:
    """
    生成 AgentRun 签名头（手写实现，无外部依赖）。
    返回包含 Agentrun-Authorization、x-acs-date、x-acs-content-sha256 等的 headers。
    content_type 若提供会参与签名（与 http-auth-acs 一致，SignedHeaders 含 content-type）。
    """
    if not access_key_id or not access_key_secret:
        raise ValueError("Access Key ID and Secret are required")

    parsed = urlparse(url)
    host = parsed.netloc
    pathname = parsed.path or "/"
    query_params: dict = {}
    if parsed.query:
        for pair in parsed.query.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                query_params[k] = v

    now = sign_time if sign_time is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date = now.strftime("%Y%m%d")

    headers_for_sign: dict = {
        "host": host,
        "x-acs-date": timestamp,
        "x-acs-content-sha256": UNSIGNED_PAYLOAD,
    }
    if security_token:
        headers_for_sign["x-acs-security-token"] = security_token
    if content_type is not None:
        headers_for_sign["content-type"] = content_type

    scope = _build_scope(date, region, product)
    canonical_request = _calc_canonical_request(
        method,
        pathname,
        query_params,
        headers_for_sign,
        UNSIGNED_PAYLOAD,
    )
    string_to_sign = _calc_string_to_sign(canonical_request)
    signature = _calc_signature(
        access_key_secret, date, region, product, string_to_sign
    )

    signed_headers_str = ";".join(_get_signed_headers(headers_for_sign))
    auth_value = (
        f"{ALGORITHM} Credential={access_key_id}/{scope},"
        f"SignedHeaders={signed_headers_str},Signature={signature}"
    )

    result = dict(headers_for_sign)
    result["Agentrun-Authorization"] = auth_value
    return result


def get_agentrun_signed_headers_with_debug(
    url: str,
    method: str = "GET",
    access_key_id: Optional[str] = None,
    access_key_secret: Optional[str] = None,
    security_token: Optional[str] = None,
    region: str = "cn-hangzhou",
    product: str = "agentrun",
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
    sign_time: Optional[datetime] = None,
) -> tuple[dict, dict]:
    """
    与 get_agentrun_signed_headers 相同，但额外返回调试信息，便于与 ram-e2e-test 或服务端对比。
    返回 (headers, debug_dict)，debug_dict 含 canonical_request, string_to_sign, signature。
    """
    if not access_key_id or not access_key_secret:
        raise ValueError("Access Key ID and Secret are required")

    parsed = urlparse(url)
    host = parsed.netloc
    pathname = parsed.path or "/"
    query_params: dict = {}
    if parsed.query:
        for pair in parsed.query.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                query_params[k] = v

    now = sign_time if sign_time is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date = now.strftime("%Y%m%d")

    headers_for_sign: dict = {
        "host": host,
        "x-acs-date": timestamp,
        "x-acs-content-sha256": UNSIGNED_PAYLOAD,
    }
    if security_token:
        headers_for_sign["x-acs-security-token"] = security_token
    if content_type is not None:
        headers_for_sign["content-type"] = content_type

    scope = _build_scope(date, region, product)
    canonical_request = _calc_canonical_request(
        method,
        pathname,
        query_params,
        headers_for_sign,
        UNSIGNED_PAYLOAD,
    )
    string_to_sign = _calc_string_to_sign(canonical_request)
    signature = _calc_signature(
        access_key_secret, date, region, product, string_to_sign
    )
    signing_key = _get_signing_key(access_key_secret, date, region, product)

    signed_headers_str = ";".join(_get_signed_headers(headers_for_sign))
    auth_value = (
        f"{ALGORITHM} Credential={access_key_id}/{scope},"
        f"SignedHeaders={signed_headers_str},Signature={signature}"
    )

    result = dict(headers_for_sign)
    result["Agentrun-Authorization"] = auth_value
    debug = {
        "canonical_request": canonical_request,
        "string_to_sign": string_to_sign,
        "signing_key_hex": signing_key.hex(),
        "signature": signature,
    }
    return result, debug
