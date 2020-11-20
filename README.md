# easyquery
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/easyquery.svg)](https://anaconda.org/conda-forge/easyquery)
[![PyPI version](https://img.shields.io/pypi/v/easyquery.svg)](https://pypi.python.org/pypi/easyquery)

Create easy-to-use Query objects that can apply on NumPy structured arrays, astropy Table, and Pandas DataFrame.

Tired of writing lots of brackets and keeping track of variable names when filtering table data? Enter `easyquery`! 

Before `easyquery`:
```python
subtable = table[table["population"] >= 20000]
subtable = subtable[subtable["population"] / subtable["area"] >= 1000]
```

With `easyquery`
```python
subtable = Query("population >= 20000", "population / area >= 1000").filter(table)
```

## Installation

You can install `easyquey` from conda-forge:

```bash
conda install scipy --channel conda-forge
```

Or from PyPI:

```bash
pip install easyquery
```

## Usage

### Creating Query objects

The most important concept that easyquery introduces is a _Query_ object,
which is an object that represents the queries (conditions) that you want to
apply to your table data.

For most simple cases a Query object can be created with a simple string:

```python
q1 = Query('population >= 20000')
q2 = Query('population / area >= 1000')
```

The string will be passed to numexpr and you can find a list of supported
operators and math functions
[here](https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-operators).

You can also combine multiple conditions at once:

```python
q3 = Query('population >= 20000', 'area < 300') # satisfies both
```

A Query object can also be created with a tuple, where the first element of
the tuple should be a callable, and the rest should be the field names that
correspond to the argument list of the callable.
This construction allows you to specify more complex queries or to use functions
that are not supported by numexpr.
For example, `q4` below has the same effect as `q2` above.

```python
q4 = Query((lambda x, y: x / y >= 1000, 'population', 'area'))
```

You can also use `QueryMaker` to create some commonly used conditions that
cannot be easily written as simple string.

```
# for string operations
q5 = QueryMaker.equals('name', 'Paris')
q6 = QueryMaker.contains('name', 'New')
q7 = QueryMaker.startswith('name', 'San')

# for checking if the column values are in another list
q8 = QueryMaker.in1d('id', [1, 3, 6, 7])
```

Query objects can be combined with `&` (and), `|` (or), `^` (xor), and cen be
modified by `~` (not). Each of these operation returns a new Query object.

```python
q9 = (~q1 | Query('established_year > 1900'))
```

### Using Query objects

A Query object has three major methods: `filter`, `count`, and `mask`.
All of them can operate on NumPy structured arrays, astropy Tables, and pandas DataFrames:

- `filter` returns a new table that only has entries satisfying the query;
- `count` returns the number of entries satisfying the query;
- `mask` returns a bool array for masking the table.

```python
import numpy as np
from easyquery import Query
t = np.array([(1, 5, 4.5), (1, 1, 6.2), (3, 2, 0.5), (5, 5, -3.5)],
             dtype=np.dtype([('a', '<i8'), ('b', '<i8'), ('c', '<f8')]))

q = Query('a > 3')
q.filter(t)
q.count(t)
q.mask(t)
```
