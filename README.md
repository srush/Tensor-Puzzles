# Tensor Puzzles
- by [Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp) 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)



When learning a tensor programming language like PyTorch or Numpy it
is tempting to rely on the standard library (or more honestly
StackOverflow) to find a magic function for everything.  But in
practice, the tensor language is extremely expressive, and you can
do most things from first principles and clever use of broadcasting.



This is a collection of 16 tensor puzzles. Like chess puzzles these are
not meant to simulate the complexity of a real program, but to practice
in a simplified environment. Each puzzle asks you to reimplement one
function in the NumPy standard library without magic. 


* [Rules](#Rules)
* [Puzzle 1 - ones](#puzzle-1---ones).
* [Puzzle 2 - sum](#puzzle-2---sum).
* [Puzzle 3 - outer](#puzzle-3---outer).
* [Puzzle 4 - diag](#puzzle-4---diag).
* [Puzzle 5 - eye](#puzzle-5---eye).
* [Puzzle 6 - triu](#puzzle-6---triu).
* [Puzzle 7 - cumsum](#puzzle-7---cumsum).
* [Puzzle 8 - diff](#puzzle-8---diff).
* [Puzzle 9 - vstack](#puzzle-9---vstack).
* [Puzzle 10 - roll](#puzzle-10---roll).
* [Puzzle 11 - flip](#puzzle-11---flip).
* [Puzzle 12 - compress](#puzzle-12---compress).
* [Puzzle 13 - pad_to](#puzzle-13---pad_to).
* [Puzzle 14 - sequence_mask](#puzzle-14---sequence_mask).
* [Puzzle 15 - bincount](#puzzle-15---bincount).
* [Puzzle 16 - scatter_add](#puzzle-16---scatter_add).


## Rules

1. Each puzzle needs to be solved in 1 line (<80 columns) of code.
2. You are allowed @, arithmetic, comparison, `shape`, any indexing (e.g. `a[:j], a[:, None], a[arange(10)]`), and previous puzzle functions.
3. Additionally you are allowed these two functions:


```python
def arange(i: int):
    "Use this function to replace a for-loop."
    return torch.tensor(range(i))

draw_examples("arange", [{str(i) : arange(i)} for i in [5, 3, 9]])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_2_0.png)
    



```python
def where(q, a, b):
    "Use this statement to replace an if-statement."
    return (q * a) + (~q) * b

# In my diagrams, orange is positive/True, where is zero/False, and green is negative.

examples = [(tensor([False]), tensor([10]), tensor([0])),
            (tensor([False, True]), tensor([1, 1]), tensor([-10, 0])),
            (tensor([False, True]), tensor([1]), tensor([-10, 0])),
            (tensor([[False, True], [True, False]]), tensor([1]), tensor([-10, 0])),
            (tensor([[False, True], [True, False]]), tensor([[0], [10]]), tensor([-10, 0])),
           ]
            

draw_examples("where", [{"q": q, "a":a, "b":b, "ret": where(q, a, b)} for q, a, b in examples])

```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_3_0.png)
    


### Anti-Rules

1. Nothing else. No `view`, `sum`, `take`, `squeeze`, `tensor`.
2. No cheating. Stackoverflow is great, but this is about first-principles.
3. Hint... these puzzles are mostly about Broadcasting. Make sure you understand this rule.

![](https://pbs.twimg.com/media/FQywor0WYAssn7Y?format=png&name=large)


```python
examples = [(arange(4), arange(5)[:, None]) ,
            (arange(3)[:, None], arange(2))]
draw_examples("broadcast", [{"a": a, "b":b, "ret": a + b} for a, b in examples])

```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_6_0.png)
    


## Running puzzles

Each example, corresponds to a unit test which will randomly
try to break your code based on the spec. The spec is written in
standard python with lists.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)

After you convince yourself it is right, uncomment the test for each example. If the test succeeds, you will get a puppy. 

## Puzzle 1 - ones

Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.


```python
def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1
        
def ones(i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'

test_ones = make_test("one", ones, ones_spec, add_sizes=["i"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_9_0.png)
    



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
        
def sum(a: TT["i"]) -> TT[1]:
    assert False, 'Not implemented yet.'


test_sum = make_test("sum", sum, sum_spec)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_12_0.png)
    



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
            
def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    assert False, 'Not implemented yet.'
    
test_outer = make_test("outer", outer, outer_spec)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_15_0.png)
    



```python
# run_test(test_outer)
```

## Puzzle 4 - diag

Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.


```python
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]
        
def diag(a: TT["i", "i"]) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_diag = make_test("diag", diag, diag_spec)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_18_0.png)
    



```python
# run_test(test_diag)
```

## Puzzle 5 - eye

Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.


```python
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1
        
def eye(j: int) -> TT["j", "j"]:
    assert False, 'Not implemented yet.'
    
test_eye = make_test("eye", eye, eye_spec, add_sizes=["j"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_21_0.png)
    



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
                
def triu(j: int) -> TT["j", "j"]:
    assert False, 'Not implemented yet.'


test_triu = make_test("triu", triu, triu_spec, add_sizes=["j"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_24_0.png)
    



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

def cumsum(a: TT["i"]) -> TT["i"]:
    assert False, 'Not implemented yet.'

test_cumsum = make_test("cumsum", cumsum, cumsum_spec)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_27_0.png)
    



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

def diff(a: TT["i"], i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'

test_diff = make_test("diff", diff, diff_spec, add_sizes=["i"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_30_0.png)
    



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

def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    assert False, 'Not implemented yet.'


test_vstack = make_test("vstack", vstack, vstack_spec)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_33_0.png)
    



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
            
def roll(a: TT["i"], i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_roll = make_test("roll", roll, roll_spec, add_sizes=["i"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_36_0.png)
    



```python
# run_test(test_roll)
```

## Puzzle 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector


```python
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]
        
def flip(a: TT["i"], i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_flip = make_test("flip", flip, flip_spec, add_sizes=["i"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_39_0.png)
    



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
            
def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_compress = make_test("compress", compress, compress_spec, add_sizes=["i"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_42_0.png)
    



```python
# run_test(test_compress)
```

## Puzzle 13 - pad_to


Compute pad_to - eliminate or add 0s to change size of vector.


```python
def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]


def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    assert False, 'Not implemented yet.'


test_pad_to = make_test("pad_to", pad_to, pad_to_spec, add_sizes=["i", "j"])
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_45_0.png)
    



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
    
def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    assert False, 'Not implemented yet.'


def constraint_set_length(d):
    d["length"] = d["length"] % d["values"].shape[1]
    return d


test_sequence = make_test("sequence_mask",
    sequence_mask, sequence_mask_spec, constraint=constraint_set_length
)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_48_0.png)
    



```python
# run_test(test_sequence)
```

## Puzzle 15 - bincount

Compute [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.


```python
def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1
        
def bincount(a: TT["i"], j: int) -> TT["j"]:
    assert False, 'Not implemented yet.'


def constraint_set_max(d):
    d["a"] = d["a"] % d["return"].shape[0]
    return d


test_bincount = make_test("bincount",
    bincount, bincount_spec, add_sizes=["j"], constraint=constraint_set_max
)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_51_0.png)
    



```python
# run_test(test_bincount)
```

## Puzzle 16 - scatter_add

Compute [scatter_add](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) - add together values that link to the same location.


```python
def scatter_add_spec(values, link, out):
    for j in range(len(values)):
        out[link[j]] += values[j]
        
def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    assert False, 'Not implemented yet.'


def constraint_set_max(d):
    d["link"] = d["link"] % d["return"].shape[0]
    return d


test_scatter_add = make_test("scatter_add",
    scatter_add, scatter_add_spec, add_sizes=["j"], constraint=constraint_set_max
)
```


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_54_0.png)
    



```python
# run_test(test_scatter_add)
```


# Speed Run Mode!

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

    ones 40
    sum 40
    outer 40
    diag 40
    eye 40
    triu 40
    cumsum 40
    diff 40
    vstack 40
    roll 40
    flip 40
    compress 40
    pad_to 40
    sequence_mask 40
    bincount 40
    scatter_add 40

