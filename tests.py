import numpy as np
from easyquery import Query

def test_valid_init():
    """
    test valid Query object creation
    """
    q1 = Query()
    q2 = Query(None)
    q3 = Query('x > 2')
    q4 = Query(lambda t: t['x'] > 2)
    q5 = Query((lambda c: c > 2, 'x'))
    q6 = Query('x > 2', lambda t: t['x'] > 2, (lambda c: c > 2, 'x'))
    q7 = Query(q3)
    q8 = Query(q3, 'x > 2')


def _check_invalid_init(*queries):
    try:
        q = Query(*queries)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_invalid_init():
    """
    test invalid Query object creation
    """
    for q in (1, [lambda x: x>1, 'a'], (lambda x: x>1,), ('a', lambda x: x>1)):
        _check_invalid_init(q)


def _gen_test_table():
    return np.array([(1, 5, 4.5), (1, 1, 6.2), (3, 2, 0.5), (5, 5, -3.5)],
                    dtype=np.dtype([('a', '<i8'), ('b', '<i8'), ('c', '<f8')]))


def _check_query_on_table(table, query_object, true_mask=None):
    if true_mask is None:
        true_mask = np.ones(len(table), np.bool)

    assert (query_object.filter(table) == table[true_mask]).all(), 'filter not correct'
    assert query_object.count(table) == np.count_nonzero(true_mask), 'count not correct'
    assert (query_object.mask(table) == true_mask).all(), 'mask not correct'


def test_simple_query():
    """
    test simple queries
    """
    t = _gen_test_table()
    _check_query_on_table(t, Query(), None)
    _check_query_on_table(t, Query('a > 3'), t['a'] > 3)
    _check_query_on_table(t, Query('a == 100'), t['a'] == 100)
    _check_query_on_table(t, Query('b > c'), t['b'] > t['c'])
    _check_query_on_table(t, Query('a < 3', 'b > c'), (t['a'] < 3) & (t['b'] > t['c']))


def test_compound_query():
    """
    test compound queries
    """
    t = _gen_test_table()
    q1 = Query('a == 1')
    m1 = t['a'] == 1
    q2 = Query('a == b')
    m2 = t['a'] == t['b']
    q3 = Query('b > c')
    m3 = t['b'] > t['c']

    q4 = ~~q3
    m4 = ~~m3
    q5 = q1 & q2 | q3
    m5 = m1 & m2 | m3
    q6 = ~q1 | q2 ^ q3
    m6 = ~m1 | m2 ^ m3
    q7 = q5 ^ q6
    m7 = m5 ^ m6
    q7 |= q2
    m7 |= m2

    _check_query_on_table(t, q1, m1)
    _check_query_on_table(t, q2, m2)
    _check_query_on_table(t, q3, m3)
    _check_query_on_table(t, q4, m4)
    _check_query_on_table(t, q5, m5)
    _check_query_on_table(t, q6, m6)
    _check_query_on_table(t, q7, m7)


if __name__ == '__main__':
    test_valid_init()
    test_invalid_init()
    test_simple_query()
    test_compound_query()
