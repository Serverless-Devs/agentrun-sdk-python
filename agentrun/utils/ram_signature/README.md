# AgentRun RAM 签名独立实现

本目录为 **独立文件夹实现**，**Python 手写签名逻辑**，无 `alibabacloud_signer_inner` 等外部签名库依赖。  
**Node.js 版本** 位于 [agentrun-sdk-nodejs](../agentrun-sdk-nodejs) 的 `src/utils/ram-signature/` 目录。

## 目录结构（Python SDK）

```
ram_signature/
├── README.md           # 本说明
├── __init__.py         # Python 包入口，导出 get_agentrun_signed_headers
└── python/             # Python 手写实现
    ├── __init__.py
    └── signer.py       # AGENTRUN4-HMAC-SHA256 纯手写
```

## 算法说明

- **算法名**: `AGENTRUN4-HMAC-SHA256`
- **Payload**: `UNSIGNED-PAYLOAD`（不参与 body 哈希）
- **参与签名的头**: `host`, `x-acs-date`, `x-acs-content-sha256`，可选 `x-acs-security-token`
- **流程**: Canonical Request → StringToSign → HMAC 派生 Key → Signature，与阿里云 OSS V4 / ACS 风格一致

## Python 使用

在 AgentRun Python SDK 内已通过 `agentrun.utils.ram_signature` 或 `agentrun.ram_signature` 使用：

```python
from agentrun.ram_signature import get_agentrun_signed_headers

headers = get_agentrun_signed_headers(
    url="https://xxx-ram.agentrun-data.cn-hangzhou.aliyuncs.com/path",
    method="GET",
    access_key_id="ak",
    access_key_secret="sk",
    region="cn-hangzhou",
    product="agentrun",
)
# headers["Agentrun-Authorization"], headers["x-acs-date"], ...
```

## Node.js 使用

Node.js 实现位于 **agentrun-sdk-nodejs** 仓库的 `src/utils/ram-signature/` 目录，通过 `@agentrun/sdk` 的 utils 导出：

```typescript
import { getAgentrunSignedHeaders } from '@agentrun/sdk';

const headers = getAgentrunSignedHeaders({
  url: 'https://xxx-ram.agentrun-data.cn-hangzhou.aliyuncs.com/path',
  method: 'GET',
  accessKeyId: 'ak',
  accessKeySecret: 'sk',
  region: 'cn-hangzhou',
  product: 'agentrun',
});
// headers['Agentrun-Authorization'], headers['x-acs-date'], ...
```

## 与 ram-e2e-test 的对应关系

逻辑与 ram-e2e-test/signature_helper.py 一致，用于替代原 `GetAccessToken` + `Agentrun-Access-Token` 的 Data API 鉴权方式；此处为手写实现，不依赖 `alibabacloud_signer_inner`。
