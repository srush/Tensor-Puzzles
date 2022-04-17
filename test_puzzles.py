g# + [markdown]
# # Tensor Puzzles
# - [Sasha Rush](http://rush-nlp.com)
#

# When starting with a tensor programming language like PyTorch or
# Numpy it is tempting to rely on the standard library (or more
# honestly stackoverflow) to find a function for everything.
# But in practice, the tensor language is extremely expressive.
# You can do most things from first principles.


# This is a collection of 15 tensor puzzles. Like chess puzzles these are
# not meant to simulate the complexity of a real program, but to practice
# in a simplified environment. Each puzzle asks you to reimplement one
# function in the NumPy standard.

# ### Rules

# 1. Each can be solved in 1 line (<80 columns) of code.
# 2. You are allowed  @, *, ==, <=, indexing, and previous puzzle functions.
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


# ### Running puzzles

# Each example, corresponds to a unit test which will randomly
# try to break your code based on the spec.

# To run these you can run with `pytest`. If you are runing in a
# notebook, just uncomment the test for each example.

# [Start at problem 1!](#puzzle-1---ones).


# ## Test Harness

# Here is the code for automatic testing (if you are interested), or you can 

# +
# !pip install torchtyping hypothesis pytest

# +

import typing
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, tuples, composite, floats
from hypothesis import given
import numpy as np
from torchtyping import TensorType


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


# + [markdown]
# ## Puzzle 1 - ones
#
# Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.

# +
def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1


# +
def ones(i: int) -> TensorType["i"]:
    assert False, 'Not implemented yet.'


test_ones = make_test(ones, ones_spec, add_sizes=["i"])
# test_ones()


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
def sum(a: TensorType["i"]) -> TensorType[1]:
    assert False, 'Not implemented yet.'


test_sum = make_test(sum, sum_spec)
# test_sum()


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
def outer(
    a: TensorType[
        "i",
    ],
    b: TensorType["j"],
) -> TensorType["i", "j"]:
    assert False, 'Not implemented yet.'


test_outer = make_test(outer, outer_spec)
# test_outer()


# + [markdown]
# ## Puzzle 4 - diag
#
# Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.

# +
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]


# +
def diag(a: TensorType["i", "i"]) -> TensorType["i"]:
    assert False, 'Not implemented yet.'


test_diag = make_test(diag, diag_spec)()
# test_diag()

# + [markdown]
# ## Puzzle 5 - eye
#
# Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.

# +
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1


# +
def eye(j: int) -> TensorType["j", "j"]:
    return where(arange(j)[:, None] == arange(j), 1, 0)


# +
test_eye = make_test(eye, eye_spec, add_sizes=["j"])
# test_eye()

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
def triu(j: int) -> TensorType["j", "j"]:
    assert False, 'Not implemented yet.'


test_triu = make_test(triu, triu_spec, add_sizes=["j"])
# test_triu()

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
def cumsum(a: TensorType["i"]) -> TensorType["i"]:
    assert False, 'Not implemented yet.'


test_cumsum = make_test(cumsum, cumsum_spec)
# test_cumsum()


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
def diff(a: TensorType["i"], i: int) -> TensorType["i"]:
    assert False, 'Not implemented yet.'


test_diff = make_test(diff, diff_spec, add_sizes=["i"])
# test_diff()

# + [markdown]
# ## Puzzle 7 - vstack
#
# Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors

# +
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]


# +
def vstack(a: TensorType["i"], b: TensorType["i"]) -> TensorType[2, "i"]:
    assert False, 'Not implemented yet.'


test_vstack = make_test(vstack, vstack_spec)()
# test_vstack()

# + [markdown]
# ## Puzzle 8 - roll
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
def roll(a: TensorType["i"], i: int) -> TensorType["i"]:
    assert False, 'Not implemented yet.'


test_roll = make_test(roll, roll_spec, add_sizes=["i"])
# test_roll()

# + [markdown]
# ## Puzzle 9 - flip
#
# Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector

# +
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]


# +
def flip(a: TensorType["i"], i: int) -> TensorType["i"]:
    assert False, 'Not implemented yet.'


test_flip = make_test(flip, flip_spec, add_sizes=["i"])
# test_flip()

# + [markdown]
# ## Puzzle 10 - compress
#
#
# Compute [compress](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - keep only masked entries (left-aligned).

# +
def compress_spec(groups, values, out):
    j = 0
    for i in range(len(groups)):
        if groups[i]:
            out[j] = values[i]
            j += 1


# +
def compress(groups: TensorType["i", bool], values: TensorType["i"]) -> TensorType["i"]:
    assert False, 'Not implemented yet.'
        groups[:, None], eye(groups.shape[0])[cumsum(groups.long()) - 1], 0
    )


test_compress = make_test(compress, compress_spec)
# test_compress()


# + [markdown]
# ## Puzzle 12 - pad_to
#
#
# Compute pad_to - eliminate or add 0s to change size of vector.


# + id="-DsZHgOTroVN"
def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]


def pad_to(a: TensorType["i"], i: int, j: int) -> TensorType["j"]:
    assert False, 'Not implemented yet.'


test_pad_to = make_test(pad_to, pad_to_spec, add_sizes=["i", "j"])
# test_pad_to()


# + [markdown]
# ## Puzzle 13 - sequence_mask
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
def sequence_mask(
    values: TensorType["i", "j"], length: TensorType["i", int]
) -> TensorType["i", "j"]:
    assert False, 'Not implemented yet.'


def constraint_set_length(d):
    d["length"] = d["length"] % d["values"].shape[0]
    return d


test_sequence = make_test(
    sequence_mask, sequence_mask_spec, constraint=constraint_set_length
)

# test_sequence()


# + [markdown]
# ## Puzzle 14: bincount
#
# Compute [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.


# +
def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1


# +
def bincount(a: TensorType["i"], j: int) -> TensorType["j"]:
    assert False, 'Not implemented yet.'


def constraint_set_max(d):
    d["a"] = d["a"] % d["return"].shape[0]
    return d


test_bincount = make_test(
    bincount, bincount_spec, add_sizes=["j"], constraint=constraint_set_max
)
# test_bincount()


# + [markdown]
# ## Puzzle 15: scatter_add
#
# Compute `scatter_add` - add togeter values that scatter together.


# +
def scatter_add_spec(values, link, out):
    for j in range(len(link)):
        out[j] += values[link[j]]


# +
def scatter_add(
    values: TensorType["i"], link: TensorType["j"], j: int
) -> TensorType["j"]:
    assert False, 'Not implemented yet.'


def constraint_set_max(d):
    d["link"] = d["link"] % d["values"].shape[0]
    return d


test_scatter_add = make_test(
    scatter_add, scatter_add_spec, add_sizes=["j"], constraint=constraint_set_max
)
# test_scatter_add()
