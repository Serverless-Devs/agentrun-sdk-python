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
