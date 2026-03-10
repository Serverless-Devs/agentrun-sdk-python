"""测试独立文件夹内的 RAM 签名手写实现（无 mock）。"""

from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest

from agentrun.utils.ram_signature.signer import (
    _canonical_headers,
    _canonical_uri,
    _percent_encode,
    get_agentrun_signed_headers,
    get_agentrun_signed_headers_with_debug,
)


class TestRamSignatureStandalone:
    """对手写实现的直接测试，不依赖 data_api 或 mock。"""

    def test_returns_required_headers(self):
        headers = get_agentrun_signed_headers(
            url="https://uid-ram.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes/s1/health",
            method="GET",
            access_key_id="ak",
            access_key_secret="sk",
            region="cn-hangzhou",
            product="agentrun",
        )
        assert "Agentrun-Authorization" in headers
        assert "x-acs-date" in headers
        assert "x-acs-content-sha256" in headers
        assert headers["x-acs-content-sha256"] == "UNSIGNED-PAYLOAD"
        assert "host" in headers
        assert (
            "uid-ram.agentrun-data.cn-hangzhou.aliyuncs.com" in headers["host"]
        )

    def test_authorization_format(self):
        headers = get_agentrun_signed_headers(
            url="https://example.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            method="GET",
            access_key_id="test-ak",
            access_key_secret="test-sk",
        )
        auth = headers["Agentrun-Authorization"]
        assert auth.startswith("AGENTRUN4-HMAC-SHA256 ")
        assert "Credential=test-ak/" in auth
        assert "SignedHeaders=" in auth
        assert "Signature=" in auth

    def test_requires_ak_sk(self):
        with pytest.raises(ValueError, match="Access Key ID and Secret"):
            get_agentrun_signed_headers(
                url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/",
                access_key_id="",
                access_key_secret="sk",
            )
        with pytest.raises(ValueError, match="Access Key ID and Secret"):
            get_agentrun_signed_headers(
                url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/",
                access_key_id="ak",
                access_key_secret="",
            )

    def test_security_token_in_headers_when_provided(self):
        headers = get_agentrun_signed_headers(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/",
            access_key_id="ak",
            access_key_secret="sk",
            security_token="sts-token",
        )
        assert "x-acs-security-token" in headers
        assert headers["x-acs-security-token"] == "sts-token"
        assert "x-acs-security-token" in headers["Agentrun-Authorization"]

    def test_deterministic_with_same_inputs(self):
        url = "https://uid-ram.agentrun-data.cn-hangzhou.aliyuncs.com/path?a=1"
        opts = dict(
            url=url,
            method="POST",
            access_key_id="ak",
            access_key_secret="sk",
            region="cn-hangzhou",
            product="agentrun",
        )
        from datetime import datetime, timezone

        t = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        h1 = get_agentrun_signed_headers(**opts, sign_time=t)
        h2 = get_agentrun_signed_headers(**opts, sign_time=t)
        assert h1["Agentrun-Authorization"] == h2["Agentrun-Authorization"]
        assert h1["x-acs-date"] == h2["x-acs-date"]

    def test_fixed_params_snapshot_matches_compare_script(self):
        """与 scripts/compare_ram_signature.py 使用相同固定参数，签名结果应与快照一致（便于与 ram-e2e-test 对比）。"""
        from datetime import datetime, timezone

        FIXED_SIGN_TIME = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)
        TEST_URL = "https://1431999136518149-ram.agentrun-data.cn-hangzhou.aliyuncs.com/sandboxes/sbx-xxx/health"
        DEFAULT_AK = "LTAI5t5opp3xMWk2B3a4gZVq"
        DEFAULT_SK = "secret"

        headers = get_agentrun_signed_headers(
            url=TEST_URL,
            method="GET",
            access_key_id=DEFAULT_AK,
            access_key_secret=DEFAULT_SK,
            region="cn-hangzhou",
            product="agentrun",
            sign_time=FIXED_SIGN_TIME,
        )
        auth = headers.get("Agentrun-Authorization", "")
        sig = (
            auth.split("Signature=")[-1].strip() if "Signature=" in auth else ""
        )
        # 快照：与 compare_ram_signature.py 同参数时的输出；若 ram-e2e-test 有 REF_Signature 可与此对比
        EXPECTED_SIGNATURE = (
            "e1479ea1aec37e55f91d82b1ccc48df2feef04184a911f91a3e3fe0e27d02610"
        )
        assert sig == EXPECTED_SIGNATURE, (
            f"固定参数下 Signature 与快照不一致: got {sig!r}, expected"
            f" {EXPECTED_SIGNATURE!r}. 可与 ram-e2e-test/print_ref_signature.py"
            " 输出对比。"
        )


# 固定时间 + query/body/header 多场景快照（与 scripts/print_ram_signature_snapshots.py / 官方包 / JS SDK 一致）
# JS/Python 等 SDK 可用相同输入校验签名一致，参见 TestRamSignatureFixedQueryBodyHeader。
FIXED_SIGN_TIME_QBH = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)
FIXED_AK = "LTAI5t5opp3xMWk2B3a4gZVq"
FIXED_SK = "secret"
BASE_URL_QBH = (
    "https://1431999136518149-ram.agentrun-data.cn-hangzhou.aliyuncs.com"
)
SCENARIO_1_EXPECTED = (
    "2de34f2c0ef3d6f0f6460f994ee3fea8c940241fcc2ff2f008a776f1be9b4dba"
)
SCENARIO_2_EXPECTED = (
    "a0d4e04405ddf83d93cf8b30c6064c1a68a298cf95ca6ff56be04f515b98ddbe"
)
SCENARIO_3_EXPECTED = (
    "fdcde808b0dc7526d8083e681ca9a64728e5c1e67e4c92735dfb2268ecc71fb2"
)
# 场景说明（供 JS SDK 对齐）:
# 1) GET {BASE_URL_QBH}/path?foo=bar&zoo= , no body, no content-type -> SCENARIO_1_EXPECTED
# 2) POST {BASE_URL_QBH}/path?foo=bar&zoo= , body=b"", no content-type -> SCENARIO_2_EXPECTED
# 3) POST {BASE_URL_QBH}/path?foo=bar&zoo= , body=b"", content-type=application/json -> SCENARIO_3_EXPECTED
# 时间统一: 2026-03-06T12:00:00Z (date=20260306), region=cn-hangzhou, product=agentrun, UNSIGNED-PAYLOAD


def _ref_signature_if_available(
    url: str,
    method: str,
    body: bytes | None,
    content_type: str | None,
    sign_time: datetime,
    ak: str,
    sk: str,
    region: str,
    product: str,
) -> str | None:
    """若已安装 alibabacloud_signer_inner，用相同参数生成参考签名，便于与官方包/JS 对齐。"""
    try:
        from alibabacloud_signer_inner import AcsV4HttpSigner, SignRequest
    except ImportError:
        return None

    class UnsignedPayloadSigner(AcsV4HttpSigner):
        ALGORITHM = "AGENTRUN4-HMAC-SHA256"

        def _hash_payload(self, payload) -> str:
            return "UNSIGNED-PAYLOAD"

    parsed = urlparse(url)
    host = parsed.netloc
    pathname = parsed.path or "/"
    query_params = {}
    if parsed.query:
        for pair in parsed.query.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                query_params[k] = v

    timestamp = sign_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    date = sign_time.strftime("%Y%m%d")
    headers = {
        "host": host,
        "x-acs-date": timestamp,
        "x-acs-content-sha256": "UNSIGNED-PAYLOAD",
    }
    if content_type is not None:
        headers["content-type"] = content_type

    payload = body if body is not None else b""
    sign_request = SignRequest(
        pathname=pathname,
        method=method.upper(),
        query_parameters=query_params,
        header_parameters=headers,
        payload=payload,
        access_key_id=ak,
        access_key_secret=sk,
        security_token=None,
        product=product,
        region=region,
        date=date,
    )
    signer = UnsignedPayloadSigner()
    auth_value = signer.sign(sign_request)
    return (
        auth_value.split("Signature=")[-1].strip()
        if "Signature=" in auth_value
        else None
    )


class TestRamSignatureFixedQueryBodyHeader:
    """固定时间 + query/body/header 多场景：与官方包（alibabacloud_signer_inner）及 JS SDK 结果一致。"""

    def _sig(
        self,
        url: str,
        method: str,
        body: bytes | None,
        content_type: str | None,
    ) -> str:
        headers = get_agentrun_signed_headers(
            url=url,
            method=method,
            access_key_id=FIXED_AK,
            access_key_secret=FIXED_SK,
            region="cn-hangzhou",
            product="agentrun",
            body=body,
            content_type=content_type,
            sign_time=FIXED_SIGN_TIME_QBH,
        )
        auth = headers.get("Agentrun-Authorization", "")
        return (
            auth.split("Signature=")[-1].strip() if "Signature=" in auth else ""
        )

    def test_scenario_1_get_query_no_body_no_content_type(self):
        """GET + query (foo=bar&zoo=)，无 body，无 content-type。"""
        url = f"{BASE_URL_QBH}/path?foo=bar&zoo="
        sig = self._sig(url, "GET", None, None)
        assert sig == SCENARIO_1_EXPECTED, f"got {sig!r}"
        ref = _ref_signature_if_available(
            url,
            "GET",
            None,
            None,
            FIXED_SIGN_TIME_QBH,
            FIXED_AK,
            FIXED_SK,
            "cn-hangzhou",
            "agentrun",
        )
        if ref is not None:
            assert sig == ref, "SDK 与官方包(ref) 签名应一致"

    def test_scenario_2_post_query_empty_body_no_content_type(self):
        """POST + query，body 空，无 content-type。"""
        url = f"{BASE_URL_QBH}/path?foo=bar&zoo="
        sig = self._sig(url, "POST", b"", None)
        assert sig == SCENARIO_2_EXPECTED, f"got {sig!r}"
        ref = _ref_signature_if_available(
            url,
            "POST",
            b"",
            None,
            FIXED_SIGN_TIME_QBH,
            FIXED_AK,
            FIXED_SK,
            "cn-hangzhou",
            "agentrun",
        )
        if ref is not None:
            assert sig == ref, "SDK 与官方包(ref) 签名应一致"

    def test_scenario_3_post_query_empty_body_content_type_json(self):
        """POST + query，body 空，content-type: application/json。"""
        url = f"{BASE_URL_QBH}/path?foo=bar&zoo="
        sig = self._sig(url, "POST", b"", "application/json")
        assert sig == SCENARIO_3_EXPECTED, f"got {sig!r}"
        ref = _ref_signature_if_available(
            url,
            "POST",
            b"",
            "application/json",
            FIXED_SIGN_TIME_QBH,
            FIXED_AK,
            FIXED_SK,
            "cn-hangzhou",
            "agentrun",
        )
        if ref is not None:
            assert sig == ref, "SDK 与官方包(ref) 签名应一致"


class TestSignerHelperFunctions:
    """测试签名辅助函数的边界情况"""

    def test_percent_encode_none(self):
        """测试 _percent_encode(None) 返回空字符串"""
        assert _percent_encode(None) == ""

    def test_percent_encode_tilde(self):
        """测试 _percent_encode 正确处理 ~ 字符"""
        assert "~" in _percent_encode("a~b")

    def test_canonical_uri_empty(self):
        """测试 _canonical_uri 空字符串返回 /"""
        assert _canonical_uri("") == "/"

    def test_canonical_uri_none(self):
        """测试 _canonical_uri None 返回 /"""
        assert _canonical_uri(None) == "/"

    def test_canonical_uri_normal(self):
        """测试 _canonical_uri 正常路径"""
        assert _canonical_uri("/path/to/resource") == "/path/to/resource"

    def test_canonical_headers_skips_none_values(self):
        """测试 _canonical_headers 跳过 value 为 None 的 header"""
        headers = {
            "host": "example.com",
            "x-acs-date": "2026-01-01T00:00:00Z",
            "x-acs-skip": None,
        }
        canon, signed = _canonical_headers(headers)
        assert "x-acs-skip" not in signed
        assert "host" in signed


class TestSignerNaiveDatetime:
    """测试 naive datetime（无时区信息）的处理"""

    def test_naive_datetime_gets_utc(self):
        """测试 naive datetime 被自动设置为 UTC"""
        naive_time = datetime(2026, 1, 1, 12, 0, 0)
        headers = get_agentrun_signed_headers(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            access_key_id="ak",
            access_key_secret="sk",
            sign_time=naive_time,
        )
        assert headers["x-acs-date"] == "2026-01-01T12:00:00Z"


class TestSignerWithDebug:
    """测试 get_agentrun_signed_headers_with_debug 函数"""

    def test_returns_headers_and_debug(self):
        """测试返回 headers 和 debug 信息"""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        headers, debug = get_agentrun_signed_headers_with_debug(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            access_key_id="ak",
            access_key_secret="sk",
            sign_time=t,
        )
        assert "Agentrun-Authorization" in headers
        assert "x-acs-date" in headers
        assert "canonical_request" in debug
        assert "string_to_sign" in debug
        assert "signing_key_hex" in debug
        assert "signature" in debug

    def test_debug_signature_matches_headers(self):
        """测试 debug 中的 signature 与 headers 中的一致"""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        headers, debug = get_agentrun_signed_headers_with_debug(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            access_key_id="ak",
            access_key_secret="sk",
            sign_time=t,
        )
        auth = headers["Agentrun-Authorization"]
        sig_in_auth = auth.split("Signature=")[-1]
        assert sig_in_auth == debug["signature"]

    def test_debug_matches_non_debug_version(self):
        """测试 debug 版本与非 debug 版本签名一致"""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        opts = dict(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path?a=1",
            method="POST",
            access_key_id="ak",
            access_key_secret="sk",
            sign_time=t,
        )
        headers_normal = get_agentrun_signed_headers(**opts)
        headers_debug, _ = get_agentrun_signed_headers_with_debug(**opts)
        assert (
            headers_normal["Agentrun-Authorization"]
            == headers_debug["Agentrun-Authorization"]
        )

    def test_debug_requires_ak_sk(self):
        """测试 debug 版本也要求 ak/sk"""
        with pytest.raises(ValueError, match="Access Key ID and Secret"):
            get_agentrun_signed_headers_with_debug(
                url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/",
                access_key_id="",
                access_key_secret="sk",
            )

    def test_debug_with_security_token(self):
        """测试 debug 版本带 security_token"""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        headers, debug = get_agentrun_signed_headers_with_debug(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            access_key_id="ak",
            access_key_secret="sk",
            security_token="sts-token",
            sign_time=t,
        )
        assert "x-acs-security-token" in headers
        assert "x-acs-security-token" in headers["Agentrun-Authorization"]

    def test_debug_with_content_type(self):
        """测试 debug 版本带 content_type"""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        headers, debug = get_agentrun_signed_headers_with_debug(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            access_key_id="ak",
            access_key_secret="sk",
            content_type="application/json",
            sign_time=t,
        )
        assert "content-type" in headers["Agentrun-Authorization"]

    def test_debug_with_query_params(self):
        """测试 debug 版本带 query 参数"""
        t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        headers, debug = get_agentrun_signed_headers_with_debug(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path?foo=bar&zoo=",
            access_key_id="ak",
            access_key_secret="sk",
            sign_time=t,
        )
        assert "foo=bar" in debug["canonical_request"]

    def test_debug_naive_datetime(self):
        """测试 debug 版本处理 naive datetime"""
        naive_time = datetime(2026, 1, 1, 12, 0, 0)
        headers, debug = get_agentrun_signed_headers_with_debug(
            url="https://x.agentrun-data.cn-hangzhou.aliyuncs.com/path",
            access_key_id="ak",
            access_key_secret="sk",
            sign_time=naive_time,
        )
        assert headers["x-acs-date"] == "2026-01-01T12:00:00Z"
