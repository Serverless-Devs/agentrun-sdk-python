"""异常定义"""

from typing import Optional


class AgentRunError(Exception):
    """AgentRun SDK 基础异常类"""

    def __init__(
        self,
        message: str,
        **kwargs,
    ):
        """初始化异常

        Args:
            message: 错误消息
            details: 详细信息
        """
        msg = message or ""
        if kwargs is not None:
            msg += self.kwargs_str(**kwargs)

        super().__init__(msg)
        self.message = message
        self.details = kwargs

    @classmethod
    def kwargs_str(cls, **kwargs) -> str:
        """获取详细信息字符串

        Returns:
            str: 详细信息字符串
        """
        if not kwargs:
            return ""
        # return ", ".join([f"{key}={value!s}" for key, value in kwargs.items()])
        import json

        return json.dumps(
            kwargs,
            ensure_ascii=False,
            default=lambda x: x.__dict__,
        )

    def details_str(self) -> str:
        return self.kwargs_str(**self.details)

    def __str__(self) -> str:
        return self.message


class HTTPError(AgentRunError):
    """HTTP 异常类"""

    def __init__(
        self,
        status_code: int,
        message: str,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        self.status_code = status_code
        self.request_id = request_id
        self.details = kwargs
        super().__init__(message, **kwargs)

    def __str__(self) -> str:
        return (
            f"HTTP {self.status_code}: {self.message}. Request ID:"
            f" {self.request_id}. Details: {self.details_str()}"
        )

    def to_resource_error(
        self, resource_type: str, resource_id: Optional[str] = ""
    ):
        if self.status_code == 404 and (
            "does not exist" in self.message or "not found" in self.message
        ):
            return ResourceNotExistError(resource_type, resource_id)
        elif (self.status_code == 400 or self.status_code == 500) and (
            "already exists" in self.message
        ):
            # TODO: ModelProxy already exists returns 500
            return ResourceAlreadyExistError(resource_type, resource_id)
        else:
            return self


class ClientError(HTTPError):
    """客户端异常类"""

    def __init__(
        self,
        status_code: int,
        message: str,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(status_code, message, request_id=request_id, **kwargs)


class ServerError(HTTPError):
    """服务端异常类"""

    def __init__(
        self, status_code: int, message: str, request_id: Optional[str] = None
    ):
        super().__init__(status_code, message, request_id=request_id)


class ResourceNotExistError(AgentRunError):
    """资源不存在异常"""

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = "",
    ):
        super().__init__(
            f"Resource {resource_type}({resource_id}) does not exist."
        )


class ResourceAlreadyExistError(AgentRunError):
    """资源已存在异常"""

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = "",
    ):
        super().__init__(
            f"Resource {resource_type}({resource_id}) already exists."
        )


class DeleteResourceError(AgentRunError):
    """删除资源异常"""

    def __init__(
        self,
        message: Optional[str] = None,
    ):
        msg = "Failed to delete resource."
        if message:
            msg += f" Reason: {message}"

        super().__init__(msg)
