from typing import Optional

from agentrun.sandbox.model import TemplateType
from agentrun.utils.config import Config
from agentrun.utils.data_api import DataAPI, ResourceType

from .sandbox import Sandbox


class CustomSandbox(Sandbox):
    """Custom Sandbox"""

    _template_type = TemplateType.CUSTOM

    def get_base_url(self, config: Optional[Config] = None):
        """Get CDP WebSocket URL for browser automation."""
        api = DataAPI(
            resource_name="",
            resource_type=ResourceType.Template,
            namespace="sandboxes",
            config=config,
        )

        return api.with_path("")
