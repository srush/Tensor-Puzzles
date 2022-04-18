# + [markdown]
# # Tensor Puzzles
# - by [Sasha Rush](http://rush-nlp.com) ( [@srush_nlp](https://twitter.com/srush_nlp) )
#

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)



# When learning a tensor programming language like PyTorch or Numpy it
# is tempting to rely on the standard library (or more honestly
# StackOverflow) to find a magic function for everything.  But in
# practice, the tensor language is extremely expressive, and you can
# do most things from first principles and clever use of broadcasting.



# This is a collection of 16 tensor puzzles. Like chess puzzles these are
# not meant to simulate the complexity of a real program, but to practice
# in a simplified environment. Each puzzle asks you to reimplement one
# function in the NumPy standard library without magic. 

# ![](https://raw.githubusercontent.com/srush/Tensor-Puzzles/main/chess.jpeg)

# * [Puzzle 1 - ones](#puzzle-1---ones).
# * [Puzzle 2 - sum](#puzzle-2---sum).
# * [Puzzle 3 - outer](#puzzle-3---outer).
# * [Puzzle 4 - diag](#puzzle-4---diag).
# * [Puzzle 5 - eye](#puzzle-5---eye).
# * [Puzzle 6 - triu](#puzzle-6---triu).
# * [Puzzle 7 - cumsum](#puzzle-7---cumsum).
# * [Puzzle 8 - diff](#puzzle-8---diff).
# * [Puzzle 9 - vstack](#puzzle-9---vstack).
# * [Puzzle 10 - roll](#puzzle-10---roll).
# * [Puzzle 11 - flip](#puzzle-11---flip).
# * [Puzzle 12 - compress](#puzzle-12---compress).
# * [Puzzle 13 - pad_to](#puzzle-13---pad_to).
# * [Puzzle 14 - sequence_mask](#puzzle-14---sequence_mask).
# * [Puzzle 15 - bincount](#puzzle-15---bincount).
# * [Puzzle 16 - scatter_add](#puzzle-16---scatter_add).


# ## Rules

# 1. Each can be solved in 1 line (<80 columns) of code.
# 2. You are allowed  @, *, ==, <=, `shape`, fancy indexing (e.g. `a[:j], a[:, None], a[arange(10)]`), and previous puzzle functions.
# 3. Additionally you are allowed these two functions:

# +
import torch


def arange(i: int):
    "Think for-loop"
    return torch.tensor(range(i))


def where(q, a, b):
    "Think if-statement"
    return (q * a) + (~q) * b


# + [markdown]

# ### Anti-Rules

# 1. Nothing else. No `view`, `sum`, `take`, `squeeze`, `tensor`.
# 2. No cheating. Stackoverflow is great, but this is about first-principles.


# ## Running puzzles

# Each example, corresponds to a unit test which will randomly
# try to break your code based on the spec. The spec is written in
# standard python with lists.

# To play, fork this repo,

# ```bash
# pip install -r requirements.txt
# pytest test_puzzles.py
# ```

# Alternatively you can play in Colab above or in a notebook on your machine.

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)

# If you are runing in a notebook, just uncomment the test for each example.
# If the test succeeds you will get a puppy. 

# [Start at Puzzle 1!](#puzzle-1---ones).



# ### Test Harness

# +
# !pip install -qqq torchtyping hypothesis pytest

# +

import typing
from torchtyping import TensorType as TT
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, tuples, composite, floats
from hypothesis import given
import numpy as np
import random


size = integers(min_value=1, max_value=5)

tensor = torch.tensor

numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}
torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


@composite
def spec(draw, x):

    names = set()
    gth = typing.get_type_hints(x)
    for k in gth:
        if not hasattr(gth[k], "__metadata__"):
            continue
        dims = gth[k].__metadata__[0]["details"][0].dims
        names.update([d.name for d in dims if isinstance(d.name, str)])
    names = list(names)
    arr = draw(tuples(*[size for _ in range(len(names))]))
    sizes = dict(zip(names, arr))
    ret = {}

    for k in gth:
        if not hasattr(gth[k], "__metadata__"):
            continue
        shape = tuple(
            [
                sizes[d.name] if isinstance(d.name, str) else d.size
                for d in gth[k].__metadata__[0]["details"][0].dims
            ]
        )
        ret[k] = draw(
            arrays(
                shape=shape,
                dtype=torch_to_numpy_dtype_dict[
                    gth[k].__metadata__[0]["details"][1].dtype
                ]
                if len(gth[k].__metadata__[0]["details"]) >= 2
                else int,
            )
        )
        ret[k][ret[k] > 1000] = 1000
        ret[k][ret[k] < -1000] = -1000
        ret[k] = np.nan_to_num(ret[k], nan=0, neginf=0, posinf=0)

    ret["return"][:] = 0
    return ret, sizes


def make_test(problem, problem_spec, add_sizes=[], constraint=lambda d: d):
    @given(spec(problem))
    def test_problem(d):
        d, sizes = d
        d = constraint(d)
        out = d["return"].tolist()
        del d["return"]
        problem_spec(*d.values(), out)
        for size in add_sizes:
            d[size] = sizes[size]

        out2 = problem(*map(tensor, d.values()))
        out = tensor(out)
        out2 = torch.broadcast_to(out2, out.shape)
        assert torch.equal(
            out, out2
        ), "Two tensors are not equal\n Spec: \n\t%s \n\t%s" % (out, out2)

    return test_problem

def run_test(fn):
    fn()
    # Generate a random puppy video if you are correct.
    print("Correct!")
    from IPython.display import HTML
    pups = [
    "2m78jPG",
    "pn1e9TO",
    "MQCIwzT",
    "udLK6FS",
    "ZNem5o3",
    "DS2IZ6K",
    "aydRUz8",
    "MVUdQYK",
    "kLvno0p",
    "wScLiVz",
    "Z0TII8i",
    "F1SChho",
    "9hRi2jN",
    "lvzRF3W",
    "fqHxOGI",
    "1xeUYme",
    "6tVqKyM",
    "CCxZ6Wr",
    "lMW0OPQ",
    "wHVpHVG",
    "Wj2PGRl",
    "HlaTE8H",
    "k5jALH0",
    "3V37Hqr",
    "Eq2uMTA",
    "Vy9JShx",
    "g9I2ZmK",
    "Nu4RH7f",
    "sWp0Dqd",
    "bRKfspn",
    "qawCMl5",
    "2F6j2B4",
    "fiJxCVA",
    "pCAIlxD",
    "zJx2skh",
    "2Gdl1u7",
    "aJJAY4c",
    "ros6RLC",
    "DKLBJh7",
    "eyxH0Wc",
    "rJEkEw4"]
    return HTML("""
    <video alt="test" controls autoplay=1>
        <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
    </video>
    """%(random.sample(pups, 1)[0]))

# + [markdown]
# ## Puzzle 1 - ones
#
# Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.

# +
def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1


# +
def ones(i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_ones = make_test(ones, ones_spec, add_sizes=["i"])
# run_test(test_ones)


# + [markdown]
# ## Puzzle 2 - sum
#
# Compute [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) - the sum of a vector.

# +
def sum_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        out[0] += a[i]


# +
def sum(a: TT["i"]) -> TT[1]:
    assert False, 'Not implemented yet.'


test_sum = make_test(sum, sum_spec)
# run_test(test_sum)


# + [markdown]
# ## Puzzle 3 - outer
#
# Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.

# +
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]


# +
def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    assert False, 'Not implemented yet.'


test_outer = make_test(outer, outer_spec)
# run_test(test_outer)


# + [markdown]
# ## Puzzle 4 - diag
#
# Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.

# +
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]


# +
def diag(a: TT["i", "i"]) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_diag = make_test(diag, diag_spec)
# run_test(test_diag)

# + [markdown]
# ## Puzzle 5 - eye
#
# Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.

# +
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1


# +
def eye(j: int) -> TT["j", "j"]:
    assert False, 'Not implemented yet.'


# +
test_eye = make_test(eye, eye_spec, add_sizes=["j"])
# run_test(test_eye)

# + [markdown]
# ## Puzzle 6 - triu
#
# Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.

# +
def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0


# +
def triu(j: int) -> TT["j", "j"]:
    assert False, 'Not implemented yet.'


test_triu = make_test(triu, triu_spec, add_sizes=["j"])
# run_test(test_triu)

# + [markdown]
# ## Puzzle 7 - cumsum
#
# Compute [cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) - the cumulative sum.

# +
def cumsum_spec(a, out):
    total = 0
    for i in range(len(out)):
        out[i] = total + a[i]
        total += a[i]


# +
def cumsum(a: TT["i"]) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_cumsum = make_test(cumsum, cumsum_spec)
# run_test(test_cumsum)


# + [markdown]
# ## Puzzle 8 - diff
#
# Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.

# +
def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]


# +
def diff(a: TT["i"], i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_diff = make_test(diff, diff_spec, add_sizes=["i"])
# run_test(test_diff)

# + [markdown]
# ## Puzzle 9 - vstack
#
# Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors

# +
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]


# +
def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    assert False, 'Not implemented yet.'


test_vstack = make_test(vstack, vstack_spec)
# run_test(test_vstack)

# + [markdown]
# ## Puzzle 10 - roll
#
# Compute [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.

# +
def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]


# +
def roll(a: TT["i"], i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_roll = make_test(roll, roll_spec, add_sizes=["i"])
# run_test(test_roll)

# + [markdown]
# ## Puzzle 11 - flip
#
# Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector

# +
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]


# +
def flip(a: TT["i"], i: int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_flip = make_test(flip, flip_spec, add_sizes=["i"])
# run_test(test_flip)

# + [markdown]
# ## Puzzle 12 - compress
#
#
# Compute [compress](https://numpy.org/doc/stable/reference/generated/numpy.compress.html) - keep only masked entries (left-aligned).

# +
def compress_spec(g, v, out):
    j = 0
    for i in range(len(g)):
        if g[i]:
            out[j] = v[i]
            j += 1


# +
def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    assert False, 'Not implemented yet.'


test_compress = make_test(compress, compress_spec, add_sizes=["i"])
# run_test(test_compress)


# + [markdown]
# ## Puzzle 13 - pad_to
#
#
# Compute pad_to - eliminate or add 0s to change size of vector.


# + id="-DsZHgOTroVN"
def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]


def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    assert False, 'Not implemented yet.'


test_pad_to = make_test(pad_to, pad_to_spec, add_sizes=["i", "j"])
# run_test(test_pad_to)


# + [markdown]
# ## Puzzle 14 - sequence_mask
#
#
# Compute [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.

# +
def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0


# +
def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    assert False, 'Not implemented yet.'


def constraint_set_length(d):
    d["length"] = d["length"] % d["values"].shape[0]
    return d


test_sequence = make_test(
    sequence_mask, sequence_mask_spec, constraint=constraint_set_length
)

# run_test(test_sequence)


# + [markdown]
# ## Puzzle 15 - bincount
#
# Compute [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.


# +
def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1


# +
def bincount(a: TT["i"], j: int) -> TT["j"]:
    assert False, 'Not implemented yet.'


def constraint_set_max(d):
    d["a"] = d["a"] % d["return"].shape[0]
    return d


test_bincount = make_test(
    bincount, bincount_spec, add_sizes=["j"], constraint=constraint_set_max
)
# run_test(test_bincount)


# + [markdown]
# ## Puzzle 16 - scatter_add
#
# Compute [scatter_add](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) - add together values that link to the same location.


# +
def scatter_add_spec(values, link, out):
    for j in range(len(values)):
        out[link[j]] += values[j]


# +
def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    assert False, 'Not implemented yet.'


def constraint_set_max(d):
    d["link"] = d["link"] % d["return"].shape[0]
    return d


test_scatter_add = make_test(
    scatter_add, scatter_add_spec, add_sizes=["j"], constraint=constraint_set_max
)
# run_test(test_scatter_add)
