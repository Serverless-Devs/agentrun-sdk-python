from typing import Optional

from agentrun.utils.model import BaseModel


def test_mask_password():
    from agentrun.utils.helper import mask_password

    assert mask_password("12345678") == "12****78"
    assert mask_password("1234567") == "12***67"
    assert mask_password("123456") == "12**56"
    assert mask_password("1234") == "1**4"
    assert mask_password("123") == "1*3"
    assert mask_password("12") == "**"
    assert mask_password("1") == "*"
    assert mask_password("") == ""


def test_merge():
    from agentrun.utils.helper import merge

    assert merge(1, 2) == 2
    assert merge(
        {"key1": "value1", "key2": {"subkey1": "subvalue1"}, "key3": 0},
        {"key2": {"subkey2": "subvalue2"}, "key3": "value3"},
    ) == {
        "key1": "value1",
        "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"},
        "key3": "value3",
    }

    from agentrun.utils.helper import merge


def test_merge_list():
    from agentrun.utils.helper import merge

    assert merge({"a": ["a", "b"]}, {"a": ["b", "c"]}) == {"a": ["b", "c"]}
    assert merge({"a": ["a", "b"]}, {"a": ["b", "c"]}, concat_list=True) == {
        "a": ["a", "b", "b", "c"]
    }

    assert merge([1, 2], [3, 4]) == [3, 4]
    assert merge([1, 2], [3, 4], concat_list=True) == [1, 2, 3, 4]
    assert merge([1, 2], [3, 4], ignore_empty_list=True) == [3, 4]

    assert merge([1, 2], []) == []
    assert merge([1, 2], [], concat_list=True) == [1, 2]
    assert merge([1, 2], [], ignore_empty_list=True) == [1, 2]


def test_merge_dict():
    from agentrun.utils.helper import merge

    assert merge(
        {"key1": "value1", "key2": "value2"},
        {"key2": "newvalue2", "key3": "newvalue3"},
    ) == {"key1": "value1", "key2": "newvalue2", "key3": "newvalue3"}

    assert merge(
        {"key1": "value1", "key2": "value2"},
        {"key2": "newvalue2", "key3": "newvalue3"},
        no_new_field=True,
    ) == {"key1": "value1", "key2": "newvalue2"}

    assert merge(
        {"key1": {"subkey1": "subvalue1"}, "key2": {"subkey2": "subvalue2"}},
        {
            "key2": {"subkey2": "newsubvalue2", "subkey3": "newsubvalue3"},
            "key3": "newvalue3",
        },
    ) == {
        "key1": {"subkey1": "subvalue1"},
        "key2": {"subkey2": "newsubvalue2", "subkey3": "newsubvalue3"},
        "key3": "newvalue3",
    }

    assert merge(
        {"key1": {"subkey1": "subvalue1"}, "key2": {"subkey2": "subvalue2"}},
        {
            "key2": {"subkey2": "newsubvalue2", "subkey3": "newsubvalue3"},
            "key3": "newvalue3",
        },
        no_new_field=True,
    ) == {"key1": {"subkey1": "subvalue1"}, "key2": {"subkey2": "newsubvalue2"}}


def test_merge_class():
    from agentrun.utils.helper import merge

    class T(BaseModel):
        a: Optional[int] = None
        b: Optional[str] = None
        c: Optional["T"] = None
        d: Optional[list] = None

    assert merge(
        T(b="2", c=T(a=3), d=[1, 2]),
        T(a=5, c=T(b="8", c=None, d=[]), d=[3, 4]),
    ) == T(a=5, b="2", c=T(a=3, b="8", c=None, d=[]), d=[3, 4])

    assert merge(
        T(b="2", c=T(a=3), d=[1, 2]),
        T(a=5, c=T(b="8", c=None, d=[]), d=[3, 4]),
        concat_list=True,
    ) == T(a=5, b="2", c=T(a=3, b="8", c=None, d=[]), d=[1, 2, 3, 4])

    assert merge(
        T(b="2", c=T(a=3), d=[1, 2]),
        T(a=5, c=T(b="8", c=None, d=[]), d=[3, 4]),
        ignore_empty_list=True,
    ) == T(a=5, b="2", c=T(a=3, b="8", c=None, d=[]), d=[3, 4])

    # class 所有字段都是存在的，因此不会被 no_new_field 影响
    assert merge(
        T(b="2", c=T(a=3), d=[1, 2]),
        T(a=5, c=T(b="8", c=None, d=[]), d=[3, 4]),
    ) == merge(
        T(b="2", c=T(a=3), d=[1, 2]),
        T(a=5, c=T(b="8", c=None, d=[]), d=[3, 4]),
        no_new_field=True,
    )


def test_merge_tuple():
    """测试 tuple 合并"""
    from agentrun.utils.helper import merge

    # 两个 tuple 应该连接
    assert merge((1, 2), (3, 4)) == (1, 2, 3, 4)

    # 空 tuple
    assert merge((1, 2), ()) == (1, 2)
    assert merge((), (3, 4)) == (3, 4)


def test_merge_set():
    """测试 set 合并"""
    from agentrun.utils.helper import merge

    # 两个 set 应该取并集
    assert merge({1, 2}, {3, 4}) == {1, 2, 3, 4}
    assert merge({1, 2}, {2, 3}) == {1, 2, 3}


def test_merge_frozenset():
    """测试 frozenset 合并"""
    from agentrun.utils.helper import merge

    # 两个 frozenset 应该取并集
    assert merge(frozenset({1, 2}), frozenset({3, 4})) == frozenset(
        {1, 2, 3, 4}
    )
    assert merge(frozenset({1, 2}), frozenset({2, 3})) == frozenset({1, 2, 3})


def test_merge_object_no_new_field():
    """测试对象合并时的 no_new_field 参数"""
    from agentrun.utils.helper import merge

    class SimpleObj:

        def __init__(self):
            self.a = 1

    obj_a = SimpleObj()
    obj_a.a = 10

    obj_b = SimpleObj()
    obj_b.a = 20
    obj_b.b = 30  # type: ignore  # new field

    # 无 no_new_field 参数时应该添加新字段
    result = merge(SimpleObj(), obj_b)
    assert hasattr(result, "b")

    # 有 no_new_field=True 时不应该添加新字段
    obj_c = SimpleObj()
    obj_c.a = 10

    obj_d = SimpleObj()
    obj_d.a = 20
    obj_d.b = 30  # type: ignore

    result2 = merge(obj_c, obj_d, no_new_field=True)
    assert not hasattr(result2, "b")
    assert result2.a == 20
