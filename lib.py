        
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, booleans, lists, composite, floats
from hypothesis import given
import numpy as np
import random
import sys
import inspect
import typing
import matplotlib.pyplot as plt

import urllib
import torch
import time
from chalk import *
import chalk
from colour import Color
from IPython.display import display, SVG
import jaxtyping

class JaxTensor:
    def __init__(self, TyArg):
        self.Ty = TyArg

    def __getitem__(self, s):
        return self.Ty[torch.Tensor, s]

# In Normal jaxtyping, a Torch Tensor array of shape (5, 6) would be written:
#   jaxtyping.Real[torch.Tensor, "5 6"]
# However, this is somewhat verbose for most of our usage, so we shorten this to:
#   Reals["5 6"]
Ints = JaxTensor(jaxtyping.Int)
Reals = JaxTensor(jaxtyping.Real)
Bools = JaxTensor(jaxtyping.Bool)


def jax_to_torch_type(ann: jaxtyping.AbstractArray) -> torch.dtype:
    """Return a torch dtype given a jaxtyping annotation."""
    # jaxtyping stores a list of acceptable dtypes.
    # Match the ~widest interpretation of a given jaxtyping.ArrayDType.
    # https://pytorch.org/docs/stable/tensor_attributes.html
    preference_list = [
        ("float32", torch.float32),  # 1.0 * torch.tensor([1,2,3]).dtype
        ("int64", torch.int64),      # torch.tensor([1,2,3]).dtype
        ("complex128", torch.complex128),
        ("complex64", torch.complex64),
        ("float64", torch.float64),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("uint64", torch.int64),
        ("int32", torch.int32),
        ("uint32", torch.int64),
        ("int16", torch.int16),
        ("uint16", torch.int32),
        ("int8", torch.int8),
        ("uint8", torch.uint8),
        ("bool", torch.bool),
        ("bool_", torch.bool),
    ]
    for (dtype_str, torch_type) in preference_list:
        if dtype_str in ann.dtypes:
            return torch_type
    print(f"I don't understand how to make a torch.tensor of jaxtyping dtype {ann}")
    assert False

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


color = [Color("red")] * 50

def color(v):
    d = rectangle(1, 1)
    if v == 0:
        return d
    elif v > 0:
        return d.fill_color(Color("orange")).fill_opacity(0.4 + 0.6 *( v / 10))
    elif v < 0:
        return d.fill_color(Color("blue")).fill_opacity(0.4 + 0.6 * ( abs(v) / 10))
    
def draw_matrix(mat):
    return vcat((hcat((color(v)
                       for j, v in enumerate(inner)))
     for i, inner in enumerate(mat)))

def grid(diagrams):
    mhs = [0] * 100
    mws = [0] * 100
    for i, row in enumerate(diagrams):
        mh = 0
        for j, col in enumerate(row):
            env = col.get_envelope()
            mhs[i] = max(env.height, mhs[i])
            mws[j] = max(mws[j], env.width)
    return vcat([hcat([col.center_xy().with_envelope(rectangle(mws[j], mhs[i]))
                       for j, col in enumerate(row)], 1.0) for i, row in enumerate(diagrams)], 1.0)
            
def draw_example(data):
    name = data["name"]
    keys = list(data["vals"][0].keys())
    # cols = [[vstrut(0)] + [vstrut(0.5) / text(f"Ex. {i}", 0.5).fill_color(Color("black")).line_width(0.0) / vstrut(0.5) for i in range(len(data["vals"]))]]
    cols = []
    for k in keys:
        mat = [(vstrut(0.5) / text(k, 0.5).fill_color(Color("black")).line_width(0.0) / vstrut(0.5))]         
        for ex in data["vals"]:
            v2 = ex[k]           
            mat.append(draw_matrix(v2))
        cols.append(mat)
    
    full = grid(cols)
    
    full = (
        vstrut(1)
        / text(name, 0.75).fill_color(Color("black")).line_width(0)
        / vstrut(1)
        / full.center_xy()
    )
    full = full.pad(1.2).center_xy()
    env = full.get_envelope()
    set_svg_height(50 * env.height)
    height = 50 * env.height
    chalk.set_svg_height(300)
    return rectangle(env.width, env.height).fill_color(Color("white")) + full
    
def draw_examples(name, examples):
    data = {"name":name,
                  "vals" :[{k: [v.tolist()] if len(v.shape) == 1 else v.tolist() 
                        for k, v in example.items()}
                        for example in examples ] }
    return draw_example(data)


@composite
def spec(draw, x, min_size=1):
    # Get the type hints.
    if sys.version_info >= (3, 9):
        gth = typing.get_type_hints(x, include_extras=True)
    else:
        gth = typing.get_type_hints(x)

    # Collect all the dimension names.
    names = {}
    for k in gth:
        if issubclass(gth[k], int):
            names[k] = None
        elif issubclass(gth[k], jaxtyping.AbstractArray):
            names = names | dict.fromkeys([d.name for d in gth[k].dims if hasattr(d, 'name')])
        else:
            print(f"Ignoring param {k} with type {gth[k]}")
    names = list(names.keys())

    # draw sizes for each dim.
    size = integers(min_value=min_size, max_value=5)
    arr = draw(arrays(shape=(len(names),), unique=True, elements=size, dtype=np.int32)).tolist()
    sizes = dict(zip(names, arr))

    # Create tensors for each size.
    ret = {}
    for k in gth:
        if not issubclass(gth[k], jaxtyping.AbstractArray):
            continue
        # jaxtyping style type
        dims = []
        for d in gth[k].dims:
            if hasattr(d, 'name'):
                dims.append(sizes[d.name])
            elif hasattr(d, 'size'):
                dims.append(d.size)
            elif hasattr(d, 'elem'):
                # This is approximately what jaxtyping does.
                # Support f-string syntax.
                # https://stackoverflow.com/a/53671539/22545467
                elem = eval(f"f'{d.elem}'", sizes.copy())
                dim = eval(elem, sizes.copy())
                dims.append(int(dim))
        shape = tuple(dims)
        torch_dtype = jax_to_torch_type(gth[k])
        np_dtype = torch_to_numpy_dtype_dict[torch_dtype]
        ret[k] = draw(
            arrays(
                shape=shape,
                dtype=np_dtype,
                elements=(booleans() if torch_dtype == torch.bool else
                          floats(min_value=-1e10, max_value=1e10, width=32) if torch_dtype.is_floating_point else
                          integers(min_value=-5, max_value=5)),
                unique=False
            )
        )
        ret[k] = np.nan_to_num(ret[k], nan=0, neginf=0, posinf=0)

    ret["return"][:] = 0
    return ret, sizes


def mk_tensor_args(fn, example_args):
    """Prepare an ordered set of example_args so that all non-scalars are torch.tensors.

    Arguments:
      fn:  the function to which the args are to be passed.
      example_args:  array like parameters.

    Returns:
      a list of arguments in torch.tensor format, suitable for passing to fn
    """
    params = list(inspect.signature(fn).parameters.values())
    args = []
    for (param, v) in zip(params, example_args.values()):
        if isinstance(param.annotation, type) and issubclass(param.annotation, (int, float)):
            args.append(v)
        elif issubclass(param.annotation, jaxtyping.AbstractArray):
            torch_dtype = jax_to_torch_type(param.annotation)
            args.append(torch.tensor(v, dtype=torch_dtype))
        else:
            args.append(torch.tensor(v))
    return args


def make_test(name, problem, problem_spec, add_sizes=[], constraint=lambda d: d):
    examples = []
    for i in range(3):
        example, sizes = spec(problem, 3).example()
        example = constraint(example)
        out = example["return"].tolist()
        del example["return"]
        problem_spec(*example.values(), out)

        for size in add_sizes:
            example[size] = sizes[size]

        yours = None
        try:
            yours = problem(*mk_tensor_args(problem, example))
            
        except NotImplementedError:
            pass
        for size in add_sizes:
            del example[size]
        example["target"] = tensor(out)
        if yours is not None:
            example["yours"] = yours 
        examples.append(example)
        
    diagram = draw_examples(name, examples)
    display(SVG(diagram._repr_svg_()))
    
    @given(spec(problem))
    def test_problem(d):
        d, sizes = d
        d = constraint(d)
        out = d["return"].tolist()
        expected_torch_return_dtype = numpy_to_torch_dtype_dict[d["return"].dtype.type]
        del d["return"]
        problem_spec(*d.values(), out)
        for size in add_sizes:
            d[size] = sizes[size]

        out2 = problem(*mk_tensor_args(problem, d))
        out = tensor(out, dtype=expected_torch_return_dtype)
        out2 = torch.broadcast_to(out2, out.shape)
        assert torch.allclose(
            out, out2, atol=1e-7
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
