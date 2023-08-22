from nynoflow.sum import Sum


def test_sum() -> None:
    mysum = Sum()
    result = mysum.sum(1, 2)
    print(result)
    assert result == 3
