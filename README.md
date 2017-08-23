# easyquery
Create easy-to-use Query objects that can apply on NumPy structured arrays, astropy Table, and Pandas DataFrame

To install, run

    pip install easyquery


A Query object has three major methods: filter, count, and mask.
All of them operate on NumPy structured array and astropy Table:

- `filter` returns a new table that only has entries satisfying the query;
- `count` returns the number of entries satisfying the query;
- `mask` returns a bool array for masking the table.

For most simple cases a Query object can be created with a numexpr string.
A Query object can also be created with a tuple, where the first element of
the tuple should be a callable, and the rest should be the field names that
correspond to the argument list of the callable. See examples below.

Query objects can be combined with & (and), | (or), ^ (xor), and cen be
modified by ~ (not). These operations return a new query object.

## Examples

```python
import numpy as np
from easyquery import Query
t = np.array([(1, 5, 4.5), (1, 1, 6.2), (3, 2, 0.5), (5, 5, -3.5)],
             dtype=np.dtype([('a', '<i8'), ('b', '<i8'), ('c', '<f8')]))

q = Query('a > 3')
q.filter(t)
q.count(t)
q.mask(t)
q2 = (~q & Query('b > c'))
q2.count(t)
```
