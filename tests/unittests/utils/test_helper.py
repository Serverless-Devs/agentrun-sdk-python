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
