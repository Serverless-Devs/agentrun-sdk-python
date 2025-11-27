"""Credential 模块 / Credential Module"""

from .api import CredentialControlAPI
from .client import CredentialClient
from .credential import Credential
from .model import (
    CredentialBasicAuth,
    CredentialConfig,
    CredentialCreateInput,
    CredentialListInput,
    CredentialUpdateInput,
    RelatedResource,
)

__all__ = [
    # base
    "Credential",
    "CredentialClient",
    "CredentialControlAPI",
    # inner model
    "CredentialBasicAuth",
    "RelatedResource",
    "CredentialConfig",
    # api model
    "CredentialCreateInput",
    "CredentialUpdateInput",
    "CredentialListInput",
]
