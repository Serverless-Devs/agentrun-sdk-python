"""MemoryCollection 使用示例 / MemoryCollection Usage Example

此示例展示如何使用 MemoryCollection 模块进行记忆集合管理，包括与 mem0ai 的集成。
This example demonstrates how to use the MemoryCollection module for memory collection management,
including integration with mem0ai.
"""

import asyncio

from agentrun.memory_collection import (
    EmbedderConfig,
    EmbedderConfigConfig,
    LLMConfig,
    LLMConfigConfig,
    MemoryCollection,
    MemoryCollectionClient,
    MemoryCollectionCreateInput,
    MemoryCollectionUpdateInput,
    NetworkConfiguration,
    VectorStoreConfig,
    VectorStoreConfigConfig,
)
from agentrun.utils.config import Config


async def main():
    """主函数 / Main function"""

    # 创建配置
    # Create configuration
    config = Config()

    # 方式一：使用 Client
    # Method 1: Using Client
    print("=== 使用 MemoryCollectionClient ===")
    client = MemoryCollectionClient(config)

    # 创建记忆集合
    # Create memory collection
    create_input = MemoryCollectionCreateInput(
        memory_collection_name="memoryCollection010901",
        description="这是一个测试",
        execution_role_arn="acs:ram::1760720386195983:role/aliyunfcdefaultrole",
        embedder_config=EmbedderConfig(
            config=EmbedderConfigConfig(model="text-embedding-v4"),
            model_service_name="bailian",
        ),
        llm_config=LLMConfig(
            config=LLMConfigConfig(model="qwen3-max"),
            model_service_name="qwen3-max",
        ),
        vector_store_config=VectorStoreConfig(
            provider="aliyun_tablestore",
            config=VectorStoreConfigConfig(
                endpoint=(
                    "https://jiuqing.cn-hangzhou.vpc.tablestore.aliyuncs.com"
                ),
                instance_name="jiuqing",
                collection_name="memories010901",
                vector_dimension=1536,
            ),
        ),
        network_configuration=NetworkConfiguration(
            vpc_id="vpc-bp1r2uvn5xactndk2jdpi",
            security_group_id="sg-bp1bsf819nni9h2upltv",
            vswitch_ids=["vsw-bp1omyfoztt6mt4h9r8jy"],
        ),
    )

    try:
        # memory_collection = await client.create_async(create_input)
        print(f"创建成功: {memory_collection.memory_collection_name}")
    except Exception as e:
        print(f"创建失败: {e}")

    # 方式二：使用高层 API
    # Method 2: Using high-level API
    print("\n=== 使用 MemoryCollection 高层 API ===")

    try:
        # memory_collection = await MemoryCollection.create_async(create_input, config=config)
        # 获取已创建的模型服务
        memory_collection = MemoryCollection.get_by_name(
            "memoryCollection010901"
        )
        print(f"获取成功: {memory_collection}")
        # # 更新记忆集合
        # # Update memory collection
        # update_input = MemoryCollectionUpdateInput(
        #     description="更新后的描述"
        # )
        # await memory_collection.update_async(update_input)
        # print("更新成功")

        # # 获取记忆集合
        # # Get memory collection
        # await memory_collection.refresh_async()
        # print(f"刷新成功: {memory_collection.description}")

        # # 列出所有记忆集合
        # # List all memory collections
        # collections = await MemoryCollection.list_all_async(config=config)
        # print(f"找到 {len(collections)} 个记忆集合")

        # # 删除记忆集合
        # # Delete memory collection
        # # await memory_collection.delete_async()
        # # print("删除成功")

    except Exception as e:
        print(f"操作失败: {e}")

    # 方式三：转换为 mem0ai Memory 客户端（需要安装 agentrun-mem0ai 依赖）
    # Method 3: Convert to mem0ai Memory client (requires mem0 dependency)
    print("\n=== 转换为 mem0ai Memory 客户端 ===")

    try:
        # 使用高层 API 的 to_mem0_memory 方法
        # Use high-level API's to_mem0_memory method
        memory = MemoryCollection.to_mem0_memory("memoryCollection010901")
        print(f"✅ 成功创建 mem0ai Memory 客户端")
        print(f"   类型: {type(memory)}")

        # 使用 mem0ai Memory 客户端进行操作
        # Use mem0ai Memory client for operations
        user_id = "user123"

        # 添加记忆
        # Add memory
        result = memory.add(
            "我喜欢吃苹果和香蕉",
            user_id=user_id,
            metadata={"category": "food"},
        )
        print(f"\n✅ 添加记忆成功:")
        for idx, res in enumerate(result.get("results", []), 1):
            print(f"   {idx}. ID: {res.get('id')}, 事件: {res.get('event')}")

        # 搜索记忆
        # Search memory
        search_results = memory.search("用户喜欢吃什么水果？", user_id=user_id)
        print(f"\n✅ 搜索记忆结果:")
        for idx, result in enumerate(search_results.get("results", []), 1):
            print(
                f"   {idx}. 内容: {result.get('memory')}, 相似度:"
                f" {result.get('score', 0):.4f}"
            )

    except ImportError as e:
        print(f"⚠️  mem0ai 未安装: {e}")
        print("   安装方法: pip install agentrun-sdk[mem0]")
    except Exception as e:
        print(f"❌ mem0ai 操作失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ 示例完成")


if __name__ == "__main__":
    asyncio.run(main())
