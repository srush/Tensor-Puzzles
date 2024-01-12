# Tensor Puzzles
- by [Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp) (with Marcos Treviso)




When learning a tensor programming language like PyTorch or Numpy it
is tempting to rely on the standard library (or more honestly
StackOverflow) to find a magic function for everything.  But in
practice, the tensor language is extremely expressive, and you can
do most things from first principles and clever use of broadcasting.


This is a collection of 21 tensor puzzles. Like chess puzzles these are
not meant to simulate the complexity of a real program, but to practice
in a simplified environment. Each puzzle asks you to reimplement one
function in the NumPy standard library without magic. 


I recommend running in Colab. Click here and copy the notebook to get start.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)

If you are interested, there is also a youtube walkthrough of the puzzles 

[![Watch the video](https://img.youtube.com/vi/SiwTAyyvt5s/default.jpg)](https://youtu.be/Hafo7hIl8MU)

```python
!pip install -qqq jaxtyping beartype hypothesis pytest git+https://github.com/chalk-diagrams/chalk
!wget -q https://github.com/srush/Tensor-Puzzles/raw/main/lib.py
```


```python
from lib import draw_examples, make_test, run_test
import torch
import numpy as np
import jaxtyping
from lib import Ints, Reals, Bools  # jaxtyping shorthand 

import beartype
Tensor = torch.Tensor
tensor = torch.tensor

# Uncommenting the following will turn on type checking of your output
# sizes, but will interfere with the line counting at the end of this
# exercise. https://github.com/google/jaxtyping/issues/160
# 
# %load_ext jaxtyping
# %jaxtyping.typechecker beartype.beartype
```

## Rules

1. These puzzles are about *broadcasting*. Know this rule.

![](https://pbs.twimg.com/media/FQywor0WYAssn7Y?format=png&name=large)

2. Each puzzle needs to be solved in 1 line (<80 columns) of code.
3. You are allowed @, arithmetic, comparison, `shape`, any indexing (e.g. `a[:j], a[:, None], a[arange(10)]`), and previous puzzle functions.
4. You are *not allowed* anything else. No `view`, `sum`, `take`, `squeeze`, `tensor`.

5. You can start with these two functions:


```python
def arange(i: int):
    "Use this function to replace a for-loop."
    return torch.tensor(range(i))

draw_examples("arange", [{"" : arange(i)} for i in [5, 3, 9]])
```




    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_7_0.svg)
    




```python
# Example of broadcasting.
examples = [(arange(4), arange(5)[:, None]) ,
            (arange(3)[:, None], arange(2))]
draw_examples("broadcast", [{"a": a, "b":b, "ret": a + b} for a, b in examples])
```




    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_8_0.svg)
    




```python
def where(q, a, b):
    "Use this function to replace an if-statement."
    return (q * a) + (~q) * b

# In diagrams, orange is positive/True, where is zero/False, and blue is negative.

examples = [(tensor([False]), tensor([10]), tensor([0])),
            (tensor([False, True]), tensor([1, 1]), tensor([-10, 0])),
            (tensor([False, True]), tensor([1]), tensor([-10, 0])),
            (tensor([[False, True], [True, False]]), tensor([1]), tensor([-10, 0])),
            (tensor([[False, True], [True, False]]), tensor([[0], [10]]), tensor([-10, 0])),
           ]
draw_examples("where", [{"q": q, "a":a, "b":b, "ret": where(q, a, b)} for q, a, b in examples])
```




    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_9_0.svg)
    



## Puzzle 1 - ones

Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.


```python
def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1

def ones(i: int) -> Ints["{i}"]:
    raise NotImplementedError

test_ones = make_test("one", ones, ones_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_11_0.svg)
    



```python
# run_test(test_ones)
```

## Puzzle 2 - sum

Compute [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) - the sum of a vector.


```python
def sum_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        out[0] += a[i]
        
def sum(a: Reals["i"]) -> Reals["1"]:
    raise NotImplementedError


test_sum = make_test("sum", sum, sum_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_14_0.svg)
    



```python
# run_test(test_sum)
```

## Puzzle 3 - outer

Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.


```python
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]
            
def outer(a: Reals["i"], b: Reals["j"]) -> Reals["i j"]:
    raise NotImplementedError
    
test_outer = make_test("outer", outer, outer_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_17_0.svg)
    



```python
# run_test(test_outer)
```

## Puzzle 4 - diag

Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.


```python
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]
        
def diag(a: Reals["i i"]) -> Reals["i"]:
    raise NotImplementedError


test_diag = make_test("diag", diag, diag_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_20_0.svg)
    



```python
# run_test(test_diag)
```

## Puzzle 5 - eye

Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.


```python
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1
        
def eye(j: int) -> Ints["{j} {j}"]:
    raise NotImplementedError
    
test_eye = make_test("eye", eye, eye_spec, add_sizes=["j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_23_0.svg)
    



```python
# run_test(test_eye)
```

## Puzzle 6 - triu

Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.


```python
def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0
                
def triu(j: int) -> Ints["{j} {j}"]:
    raise NotImplementedError

test_triu = make_test("triu", triu, triu_spec, add_sizes=["j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_26_0.svg)
    



```python
# run_test(test_triu)
```

## Puzzle 7 - cumsum

Compute [cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) - the cumulative sum.


```python
def cumsum_spec(a, out):
    total = 0
    for i in range(len(out)):
        out[i] = total + a[i]
        total += a[i]

def cumsum(a: Reals["i"]) -> Reals["i"]:
    raise NotImplementedError

test_cumsum = make_test("cumsum", cumsum, cumsum_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_29_0.svg)
    



```python
# run_test(test_cumsum)
```

## Puzzle 8 - diff

Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.


```python
def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]

def diff(a: Reals["i"], i: int) -> Reals["i"]:
    assert(i == a.shape[0])
    raise NotImplementedError

test_diff = make_test("diff", diff, diff_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_32_0.svg)
    



```python
# run_test(test_diff)
```

## Puzzle 9 - vstack

Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors


```python
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]

def vstack(a: Reals["i"], b: Reals["i"]) -> Reals["2 i"]:
    raise NotImplementedError

test_vstack = make_test("vstack", vstack, vstack_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_35_0.svg)
    



```python
# run_test(test_vstack)
```

## Puzzle 10 - roll

Compute [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.


```python
def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]
            
def roll(a: Reals["i"], i: int) -> Reals["i"]:
    assert(i == a.shape[0])
    raise NotImplementedError


test_roll = make_test("roll", roll, roll_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_38_0.svg)
    



```python
# run_test(test_roll)
```

## Puzzle 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector


```python
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]
        
def flip(a: Reals["i"], i: int) -> Reals["i"]:
    assert(i == a.shape[0])
    raise NotImplementedError

test_flip = make_test("flip", flip, flip_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_41_0.svg)
    



```python
# run_test(test_flip)
```

## Puzzle 12 - compress


Compute [compress](https://numpy.org/doc/stable/reference/generated/numpy.compress.html) - keep only masked entries (left-aligned).


```python
def compress_spec(g, v, out):
    j = 0
    for i in range(len(g)):
        if g[i]:
            out[j] = v[i]
            j += 1
            
def compress(g: Bools["i"], v: Reals["i"], i: int) -> Reals["i"]:
    assert(i == v.shape[0])
    raise NotImplementedError

test_compress = make_test("compress", compress, compress_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_44_0.svg)
    



```python
# run_test(test_compress)
```

## Puzzle 13 - pad_to


Compute pad_to - eliminate or add 0s to change size of vector.


```python
def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]

def pad_to(a: Reals["i"], i: int, j: int) -> Reals["{j}"]:
    assert(i == a.shape[0])
    raise NotImplementedError

test_pad_to = make_test("pad_to", pad_to, pad_to_spec, add_sizes=["i", "j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_47_0.svg)
    



```python
# run_test(test_pad_to)
```

## Puzzle 14 - sequence_mask


Compute [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.


```python
def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0
    
def sequence_mask(values: Reals["i j"], length: Ints["i"]) -> Reals["i j"]:
    raise NotImplementedError

def constraint_set_length(d):
    d["length"] = d["length"] % d["values"].shape[1]
    return d

test_sequence = make_test("sequence_mask",
    sequence_mask, sequence_mask_spec, constraint=constraint_set_length
)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_50_0.svg)
    



```python
# run_test(test_sequence)
```

## Puzzle 15 - bincount

Compute [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.


```python
def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1
        
def bincount(a: Ints["i"], j: int) -> Ints["{j}"]:
    assert j >= max(a)
    assert all(x >= 0 for x in a)
    raise NotImplementedError

def constraint_set_max(d):
    d["a"] = d["a"] % d["return"].shape[0]
    return d


test_bincount = make_test("bincount",
    bincount, bincount_spec, add_sizes=["j"], constraint=constraint_set_max
)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_53_0.svg)
    



```python
# run_test(test_bincount)
```

## Puzzle 16 - scatter_add

Compute [scatter_add](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) - add together values that link to the same location.


```python
def scatter_add_spec(values, link, out):
    for j in range(len(values)):
        out[link[j]] += values[j]
        
def scatter_add(values: Reals["i"], link: Ints["i"], j: int) -> Reals["{j}"]:
    raise NotImplementedError

def constraint_set_max(d):
    d["link"] = d["link"] % d["return"].shape[0]
    return d

test_scatter_add = make_test("scatter_add",
    scatter_add, scatter_add_spec, add_sizes=["j"], constraint=constraint_set_max
)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_56_0.svg)
    



```python
# run_test(test_scatter_add)
```

## Puzzle 17 - flatten

Compute [flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)


```python
def flatten_spec(a, out):
    k = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            out[k] = a[i][j]
            k += 1

def flatten(a: Reals["i j"], i:int, j:int) -> Reals["i*j"]:
    assert(i == a.shape[0])
    assert(j == a.shape[1])
    raise NotImplementedError

test_flatten = make_test("flatten", flatten, flatten_spec, add_sizes=["i", "j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_59_0.svg)
    



```python
# run_test(test_flatten)
```

## Puzzle 18 - linspace

Compute [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)


```python
def linspace_spec(i, j, out):
    for k in range(len(out)):
        out[k] = float(i + (j - i) * k / max(1, len(out) - 1))

def linspace(i: Reals["1"], j: Reals["1"], n: int) -> Reals["{n}"]:
    raise NotImplementedError

test_linspace = make_test("linspace", linspace, linspace_spec, add_sizes=["n"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_62_0.svg)
    



```python
# run_test(test_linspace)
```

## Puzzle 19 - heaviside

Compute [heaviside](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html)


```python
def heaviside_spec(a, b, out):
    for k in range(len(out)):
        if a[k] == 0:
            out[k] = b[k]
        else:
            out[k] = int(a[k] > 0)

def heaviside(a: Reals["i"], b: Reals["i"]) -> Reals["i"]:
    raise NotImplementedError

test_heaviside = make_test("heaviside", heaviside, heaviside_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_65_0.svg)
    



```python
# run_test(test_heaviside)
```

## Puzzle 20 - repeat (1d)

Compute [repeat](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html)


```python
def repeat_spec(a, d, out):
    for i in range(d[0]):
        for k in range(len(a)):
            out[i][k] = a[k]
            
def repeat(a: Reals["i"], d: int) -> Reals["{d} i"]:
    raise NotImplementedError

test_repeat = make_test("repeat", repeat, repeat_spec, add_sizes=['d'])
```
    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_68_0.svg)

## Puzzle 21 - bucketize

Compute [bucketize](https://pytorch.org/docs/stable/generated/torch.bucketize.html)

```python
def bucketize_spec(v, boundaries, out):
    for i, val in enumerate(v):
        out[i] = 0
        for j in range(len(boundaries)-1):
            if val >= boundaries[j]:
                out[i] = j + 1
        if val >= boundaries[-1]:
            out[i] = len(boundaries)


def constraint_set(d):
    d["boundaries"] = np.abs(d["boundaries"]).cumsum()
    return d

def bucketize(v: Reals["i"], boundaries: Reals["j"]) -> Ints["i"]:
    raise NotImplementedError

test_bucketize = make_test("bucketize", bucketize, bucketize_spec,
                           constraint=constraint_set)
```
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_69_0.svg)

## Speed Run Mode!
What is the smallest you can make each of these?

```python
import inspect
fns = (ones, sum, outer, diag, eye, triu, cumsum, diff, vstack, roll, flip,
       compress, pad_to, sequence_mask, bincount, scatter_add)

for fn in fns:
    lines = [l for l in inspect.getsource(fn).split("\n") if not l.strip().startswith("#")]
    
    if len(lines) > 3:
        print(fn.__name__, len(lines[2]), "(more than 1 line)")
    else:
        print(fn.__name__, len(lines[1]))
```

    ones 29
    sum 29
    outer 29
    diag 29
    eye 29
    triu 29
    cumsum 29
    diff 29
    vstack 29
    roll 29
    flip 29
    compress 29
    pad_to 29
    sequence_mask 29
    bincount 29
    scatter_add 29

