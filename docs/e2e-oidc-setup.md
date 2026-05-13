# E2E Tests: OIDC Keyless Authentication Setup

This guide explains how to configure GitHub Actions OIDC (OpenID Connect) for
running E2E tests against Alibaba Cloud without storing long-lived AK/SK credentials.

## How It Works

```
GitHub Actions  ──OIDC token──>  Alibaba Cloud RAM (OIDC Provider)
                                        │
                                        ▼
                                 STS AssumeRoleWithOIDC
                                        │
                                        ▼
                              Temporary AK/SK/SecurityToken
                                        │
                                        ▼
                               E2E tests use temp creds
```

1. GitHub Actions generates an OIDC token containing repository and workflow metadata.
2. The `aliyun/configure-aliyun-credentials-action` exchanges that token with Alibaba Cloud STS.
3. STS returns temporary credentials (valid for 1 hour) scoped to a specific RAM Role.
4. E2E tests use those credentials — no permanent secrets stored anywhere.

## Prerequisites

- Alibaba Cloud account with RAM admin access
- GitHub repository admin access (to configure secrets)

## Step 1: Create OIDC Identity Provider in RAM

1. Go to [RAM Console > SSO Management > OIDC](https://ram.console.aliyun.com/providers/oidc)
2. Click **Create OIDC Provider**
3. Fill in the form:

| Field | Value |
|-------|-------|
| Provider Name | `github-actions` |
| Issuer URL | `https://token.actions.githubusercontent.com` |
| Client ID (Audience) | `sts.aliyuncs.com` |
| Description | GitHub Actions OIDC for agentrun-sdk e2e tests |

4. Click **OK** to create.
5. Copy the **Provider ARN**, it looks like:
   ```
   acs:ram::<ACCOUNT_ID>:oidc-provider/github-actions
   ```

## Step 2: Create a RAM Role for E2E Tests

1. Go to [RAM Console > Identities > Roles](https://ram.console.aliyun.com/roles)
2. Click **Create Role** > **IdP** > **OIDC**
3. Fill in the form:

| Field | Value |
|-------|-------|
| Role Name | `github-actions-e2e` |
| Select OIDC Provider | `github-actions` (created in Step 1) |
| Condition | See trust policy below |

4. Use this **Trust Policy** (edit the role after creation if the console doesn't allow full customization):

```json
{
  "Statement": [
    {
      "Action": "sts:AssumeRoleWithOIDC",
      "Condition": {
        "StringEquals": {
          "oidc:aud": "sts.aliyuncs.com"
        },
        "StringLike": {
          "oidc:sub": "repo:Serverless-Devs/agentrun-sdk-python:*"
        }
      },
      "Effect": "Allow",
      "Principal": {
        "Federated": [
          "acs:ram::<ACCOUNT_ID>:oidc-provider/github-actions"
        ]
      }
    }
  ],
  "Version": "1"
}
```

> Replace `<ACCOUNT_ID>` with your Alibaba Cloud account ID.
>
> The `oidc:sub` condition restricts access to this specific repository.
> You can narrow it further: `repo:Serverless-Devs/agentrun-sdk-python:ref:refs/heads/main`
> would limit to the main branch only.

5. Copy the **Role ARN**, it looks like:
   ```
   acs:ram::<ACCOUNT_ID>:role/github-actions-e2e
   ```

## Step 3: Attach Permission Policy to the Role

The RAM Role needs permissions to call AgentRun / Function Compute APIs
used by the E2E tests. Create a custom policy or attach existing ones:

**Recommended minimum permissions:**
- `AliyunFCReadOnlyAccess` (read-only FC access for test verification)
- A custom policy for AgentRun API operations used in tests

Example custom policy:
```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "fc:*"
      ],
      "Resource": [
        "acs:fc:<REGION>:<ACCOUNT_ID>:*"
      ]
    }
  ]
}
```

> Adjust the `Action` and `Resource` scope based on what your E2E tests actually call.
> Principle of least privilege: grant only what the tests need.

## Step 4: Configure GitHub Secrets

Go to **GitHub repo > Settings > Secrets and variables > Actions** and add:

### OIDC Secrets (required for authentication)

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `ALIBABA_CLOUD_OIDC_PROVIDER_ARN` | OIDC Provider ARN from Step 1 | `acs:ram::1234567890:oidc-provider/github-actions` |
| `ALIBABA_CLOUD_OIDC_ROLE_ARN` | RAM Role ARN from Step 2 | `acs:ram::1234567890:role/github-actions-e2e` |

### Test Configuration Secrets (required for e2e tests)

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `AGENTRUN_ACCOUNT_ID` | Alibaba Cloud account ID | `1234567890` |
| `AGENTRUN_REGION` | Region for test resources | `cn-hangzhou` |
| `AGENTRUN_CONTROL_ENDPOINT` | AgentRun control API endpoint | `https://agentrun.cn-hangzhou.aliyuncs.com` |
| `AGENTRUN_DATA_ENDPOINT` | AgentRun data API endpoint | `https://1234567890.agentrun-data.cn-hangzhou.aliyuncs.com` |


> `API_KEY` and `AGENTRUN_TEST_WORKSPACE_ID` use hardcoded placeholders in the
> workflow and do not need to be configured as secrets.

## Step 5: Verify

1. Push a commit to `main` or trigger the workflow manually via **Actions > E2E Tests > Run workflow**.
2. Check the workflow run — the "Configure Alibaba Cloud credentials (OIDC)" step should succeed.
3. E2E tests should run with temporary credentials.

## Troubleshooting

### "AssumeRoleWithOIDC failed"
- Verify the OIDC Provider issuer URL is exactly `https://token.actions.githubusercontent.com`
- Verify the Role trust policy `oidc:sub` matches your repo: `repo:Serverless-Devs/agentrun-sdk-python:*`
- Check the audience is `sts.aliyuncs.com`

### "Permission denied" during tests
- The RAM Role needs the right policies attached (Step 3)
- Check if the region in the policy matches `AGENTRUN_REGION`

### E2E tests skip on fork PRs
- This is intentional: fork PRs cannot access OIDC secrets
- Only PRs from the same repo, pushes to main, and manual triggers run e2e
