"""This is my test."""

from nynoflow.sum import Sum


def test_sum() -> None:
    """This is my test sum function."""
    mysum = Sum()
    result = mysum.sum(1, 2)
    print(result)
    assert result == 3
