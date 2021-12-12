"""
Create easy-to-use Query objects that can apply on
NumPy structured arrays, astropy Table, and Pandas DataFrame.
Project website: https://github.com/yymao/easyquery
The MIT License (MIT)
Copyright (c) 2017-2021 Yao-Yuan Mao (yymao)
http://opensource.org/licenses/MIT
"""

import warnings
import functools
import numpy as np
import numexpr as ne

__all__ = ['Query', 'QueryMaker']
__version__ = '0.4.0'


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


class Query(object):
    """
    Create a Query object, which stores the query to be apply on a table.

    A Query object has three major methods: filter, count, and mask.
    All of them operate on NumPy structured array and astropy Table:
    - `filter` returns a new table that only has entries satisfying the query;
    - `split` returns two new tables that has entries satisfying and not satisfying the query, respectively;
    - `count` returns the number of entries satisfying the query;
    - `mask` returns a bool array for masking the table;
    - `where` returns a int array for the indices that select satisfying entries.

    For most simple cases a Query object can be created with a numexpr string.
    A Query object can also be created with a tuple, where the first element of
    the tuple should be a callable, and the rest should be the field names that
    correspond to the argument list of the callable. See examples below.

    Query objects can be combined with & (and), | (or), ^ (xor), and cen be
    modified by ~ (not). These operations return a new query object.

    Examples
    --------

    >>> import numpy as np
    >>> from easyquery import Query
    >>> t = np.array([(1, 5, 4.5), (1, 1, 6.2), (3, 2, 0.5), (5, 5, -3.5)],
    ...              dtype=np.dtype([('a', '<i8'), ('b', '<i8'), ('c', '<f8')]))
    >>> t[t['a']>3]
    array([(5, 5, -3.5)], dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<f8')])

    >>> q = Query('a > 3')
    >>> q.filter(t)
    array([(5, 5, -3.5)], dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<f8')])
    >>> q.count(t)
    1
    >>> q.mask(t)
    array([False, False, False,  True], dtype=bool)
    >>> q.where(t)
    array([3], dtype=int64)

    >>> q2 = (~q & Query('b > c'))
    >>> q2.count(t)
    2

    """
    # pylint: disable=protected-access

    def __init__(self, *queries):
        self._operator = None
        self._operands = None
        self._variable_names = None
        self._query_class = type(self)

        if len(queries) == 1:
            query = queries[0]
            if isinstance(query, self._query_class):
                self._operator = query._operator
                self._operands = query._operands if query._operator is None else query._operands.copy()
            else:
                if not self._check_basic_query(query):
                    raise ValueError('Not a valid query.')
                self._operands = query

        elif len(queries) > 1:
            self._operator = 'AND'
            self._operands = [self._query_class(query) for query in queries]

    @staticmethod
    def _get_table_dict(table):
        return table

    @staticmethod
    def _get_table_len(table):
        return len(table)

    @staticmethod
    def _get_table_column(table, column):
        return table[column]

    @staticmethod
    def _mask_table(table, mask_):
        return table[mask_]

    def _combine_queries(self, other, operator, out=None):
        if operator not in {'AND', 'OR', 'XOR'}:
            raise ValueError('`operator` must be "AND" or "OR" or "XOR"')

        if not isinstance(other, self._query_class):
            other = self._query_class(other)

        if out is None:
            out = self._query_class()

        out._operator = operator

        if self._operator == operator and other._operator == operator:
            out._operands = self._operands + other._operands

        elif self._operator == operator and other._operator != operator:
            out._operands = self._operands + list((other,))

        elif self._operator != operator and other._operator == operator:
            out._operands = list((self,)) + other._operands

        else:
            out._operands = list((self, other))

        return out

    def __and__(self, other):
        return self._combine_queries(other, 'AND')

    def __iand__(self, other):
        self.copy()._combine_queries(other, 'AND', out=self)
        return self

    def __or__(self, other):
        return self._combine_queries(other, 'OR')

    def __ior__(self, other):
        self.copy()._combine_queries(other, 'OR', out=self)
        return self

    def __xor__(self, other):
        return self._combine_queries(other, 'XOR')

    def __ixor__(self, other):
        self.copy()._combine_queries(other, 'XOR', out=self)
        return self

    def __invert__(self):
        if self._operator == 'NOT':
            return self._operands.copy()
        else:
            out = self._query_class()
            out._operator = 'NOT'
            out._operands = self
        return out

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    @staticmethod
    def _check_basic_query(basic_query):
        return (
            basic_query is None or
            _is_string_like(basic_query) or
            callable(basic_query) or
            (
                isinstance(basic_query, tuple) and
                len(basic_query) > 1 and
                callable(basic_query[0])
            )
        )

    def _create_mask(self, table, basic_query):
        if _is_string_like(basic_query):
            return ne.evaluate(
                basic_query,
                local_dict=self._get_table_dict(table),
                global_dict={}
            )

        elif callable(basic_query):
            return basic_query(table)

        elif isinstance(basic_query, tuple) and len(basic_query) > 1 and callable(basic_query[0]):
            return basic_query[0](*(self._get_table_column(table, c) for c in basic_query[1:]))

    def mask(self, table):
        """
        Use the current Query object to create a mask (a boolean array)
        for `table`. Values in the returned mask are determined based on
        whether the corresponding rows satisfy input queries.

        Parameters
        ----------
        table : NumPy structured array, astropy Table, etc.

        Returns
        -------
        mask : numpy bool array
        """
        if self._operator is None:
            if self._operands is None:
                return np.ones(self._get_table_len(table), dtype=bool)
            else:
                return self._create_mask(table, self._operands)

        if self._operator == 'NOT':
            return ~self._operands.mask(table)

        if self._operator == 'AND':
            op_func = np.logical_and
        elif self._operator == 'OR':
            op_func = np.logical_or
        elif self._operator == 'XOR':
            op_func = np.logical_xor

        mask_this = self._operands[0].mask(table)
        for op in self._operands[1:]:
            mask_this = op_func(mask_this, op.mask(table), out=mask_this)

        return mask_this

    def filter(self, table, column_slice=None):
        """
        Use the current Query object to select the rows in `table`
        that satisfy input queries.
        If `column_slice` is provided, also select on columns.

        Equivalent to table[Query(...).mask(table)][column_slice]
        but with more efficient implementation.

        Parameters
        ----------
        table : NumPy structured array, astropy Table, etc.
        column_slice : Column to return. Default is None (return all columns).

        Returns
        -------
        table : filtered table
        """
        if self._operator is None and self._operands is None:
            return table if column_slice is None else self._get_table_column(table, column_slice)

        if self._operator == 'AND' and column_slice is None:
            for op in self._operands:
                table = op.filter(table)
            return table

        return self._mask_table(
            table if column_slice is None else self._get_table_column(table, column_slice),
            self.mask(table)
        )

    __call__ = filter

    def count(self, table):
        """
        Use the current Query object to count the number of rows in `table`
        that satisfy input queries.

        Equivalent to np.count_nonzero(Query(...).mask(table)).

        Parameters
        ----------
        table : NumPy structured array, astropy Table, etc.

        Returns
        -------
        count : int
        """
        if self._operator is None and self._operands is None:
            return self._get_table_len(table)

        return np.count_nonzero(self.mask(table))

    def where(self, table):
        """
        Return the indices of the rows in `table` that satisfy input queries.
        Equivalent to calling `np.flatnonzero(Query(...).mask(table)`.

        Parameters
        ----------
        table : NumPy structured array, astropy Table, etc.

        Returns
        -------
        indices : numpy int array
        """
        if self._operator is None and self._operands is None:
            return np.arange(self._get_table_len(table))

        return np.flatnonzero(self.mask(table))

    def split(self, table, column_slice=None):
        """
        Split the `table` into two parts: satisfying and not satisfy the queries.
        The function will return q.filter(table), (~q).filter(table)
        where `q` is the current Query object.

        Parameters
        ----------
        table : NumPy structured array, astropy Table, etc.

        Returns
        -------
        table_true : filtered table, satisfying the queries
        table_false : filtered table, not satisfying the queries
        """
        mask = self.mask(table)
        if column_slice is not None:
            table = self._get_table_column(table, column_slice)
        return self._mask_table(table, mask), self._mask_table(table, ~mask)

    def copy(self):
        """
        Create a copy of the current Query object.

        Returns
        -------
        out : Query object
        """
        out = self._query_class()
        out._operator = self._operator
        out._operands = self._operands if self._operator is None else self._operands.copy()
        return out

    @staticmethod
    def _get_variable_names(basic_query):
        if _is_string_like(basic_query):
            return tuple(set(ne.necompiler.precompile(basic_query)[-1]))

        elif callable(basic_query):
            warnings.warn('`variable_names` does not support a single callable query')
            return tuple()

        elif isinstance(basic_query, tuple) and len(basic_query) > 1 and callable(basic_query[0]):
            return tuple(set(basic_query[1:]))

    @property
    def variable_names(self):
        """
        Get all variable names required for this query
        """
        if self._variable_names is None:

            if self._operator is None:
                if self._operands is None:
                    self._variable_names = tuple()
                else:
                    self._variable_names = self._get_variable_names(self._operands)

            elif self._operator == 'NOT':
                self._variable_names = self._operands.variable_names

            else:
                v = list()
                for op in self._operands:
                    v.extend(op.variable_names)
                self._variable_names = tuple(set(v))

        return self._variable_names


_query_class = Query


def set_query_class(query_class=Query):
    """
    Set default query class
    """
    if not issubclass(query_class, Query):
        raise ValueError('`query_class` must be a subclass of `Query`')
    global _query_class
    _query_class = query_class


def filter(table, *queries):  # pylint: disable=redefined-builtin
    """
    A convenient function to filter `table` with `queries`.
    Equivalent to Query(*queries).filter(table)

    Parameters
    ----------
    table : NumPy structured array, astropy Table, etc.
    queries : string, tuple, callable

    Returns
    -------
    table : filtered table
    """
    return _query_class(*queries).filter(table)


def count(table, *queries):
    """
    A convenient function to count the number of entries in `table`
    that satisfy `queries`.
    Equivalent to `Query(*queries).count(table)`

    Parameters
    ----------
    table : NumPy structured array, astropy Table, etc.
    queries : string, tuple, callable

    Returns
    -------
    count : int
    """
    return _query_class(*queries).count(table)


def mask(table, *queries):
    """
    A convenient function to create a mask (a boolean array) for `table`
    given `queries`.
    Equivalent to `Query(*queries).mask(table)`

    Parameters
    ----------
    table : NumPy structured array, astropy Table, etc.
    queries : string, tuple, callable

    Returns
    -------
    mask : numpy bool array
    """
    return _query_class(*queries).mask(table)


def where(table, *queries):
    """
    A convenient function to get the indices of the rows in `table` that
    satisfy input `queries`.
    Equivalent to `Query(*queries).where(table)`

    Parameters
    ----------
    table : NumPy structured array, astropy Table, etc.
    queries : string, tuple, callable

    Returns
    -------
    indices : numpy int array
    """
    return _query_class(*queries).where(table)


def split(table, *queries):
    """
    A convenient function to split `table` into satisfying and non-satisfying parts.
    Equivalent to `Query(*queries).split(table)`

    Parameters
    ----------
    table : NumPy structured array, astropy Table, etc.
    queries : string, tuple, callable

    Returns
    -------
    table_true : filtered table, satisfying the queries
    table_false : filtered table, not satisfying the queries
    """
    return _query_class(*queries).split(table)


class QueryMaker():
    """
    provides convenience functions to generate query objects
    """
    @staticmethod
    def in1d(col_name, test_elements, assume_unique=False, invert=False):
        return _query_class((functools.partial(np.in1d, ar2=test_elements, assume_unique=assume_unique, invert=invert), col_name))

    @staticmethod
    def isin(col_name, test_elements, assume_unique=False, invert=False):
        return _query_class((functools.partial(np.isin, test_elements=test_elements, assume_unique=assume_unique, invert=invert), col_name))

    @staticmethod
    def vectorize(row_function, *col_names):
        return _query_class((lambda *args: np.fromiter(map(row_function, *args), bool),) + tuple(col_names))

    @staticmethod
    def contains(col_name, test_value):
        return QueryMaker.vectorize((lambda x: test_value in x), col_name)

    @staticmethod
    def find(col_name, test_value, start=0, end=None):
        return _query_class((lambda x: np.char.find(x, test_value, start=start, end=end) > -1, col_name))

    contains_str = find

    @staticmethod
    def equal(col_name, test_value):
        return _query_class((lambda x: x == test_value, col_name))

    equals = equal

    @staticmethod
    def not_equal(col_name, test_value):
        return _query_class((lambda x: x != test_value, col_name))

    @staticmethod
    def equal_columns(col1_name, col2_name):
        return _query_class((lambda x, y: x == y, col1_name, col2_name))

    @staticmethod
    def not_equal_columns(col1_name, col2_name):
        return _query_class((lambda x, y: x != y, col1_name, col2_name))

    @staticmethod
    def startswith(col_name, prefix, start=0, end=None):
        return _query_class((functools.partial(np.char.startswith, prefix=prefix, start=start, end=end), col_name))

    @staticmethod
    def endswith(col_name, suffix, start=0, end=None):
        return _query_class((functools.partial(np.char.endswith, suffix=suffix, start=start, end=end), col_name))

    @staticmethod
    def isfinite(col_name):
        return QueryMaker.vectorize(np.isfinite, col_name)

    @staticmethod
    def isnan(col_name):
        return QueryMaker.vectorize(np.isnan, col_name)

    @staticmethod
    def isnotnan(col_name):
        return ~QueryMaker.isnan(col_name)

    @staticmethod
    def isclose(col1_name, col2_name):
        return QueryMaker.vectorize(np.isclose, col1_name, col2_name)

    @staticmethod
    def reduce_compare(columns, reduce_func, compare_func, compare_value):
        """
        returns Query((compare_func(reduce_func(np.stack(arrays), axis=0), compare_value), *columns))
        """
        def _func(*arrays, reduce_func=reduce_func, compare_func=compare_func, compare_value=compare_value):
            return compare_func(reduce_func(np.stack(arrays), axis=0), compare_value)
        return Query((_func,) + tuple(columns))
