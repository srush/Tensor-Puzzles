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


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_2_0.png)
    



    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_3_0.png)
    


### Anti-Rules

1. Nothing else. No `view`, `sum`, `take`, `squeeze`, `tensor`.
2. No cheating. Stackoverflow is great, but this is about first-principles.
3. Hint... these puzzles are mostly about Broadcasting. Make sure you understand this rule.

![](https://pbs.twimg.com/media/FQywor0WYAssn7Y?format=png&name=large)


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_6_0.png)
    


## Running puzzles

Each example, corresponds to a unit test which will randomly
try to break your code based on the spec. The spec is written in
standard python with lists.

To play, fork this repo,

```bash
pip install -r requirements.txt
pytest test_puzzles.py
```

Alternatively you can play in Colab above or in a notebook on your machine.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)

If you are runing in a notebook, just uncomment the test for each example.
If the test succeeds you will get a puppy. 

[Start at Puzzle 1!](#puzzle-1---ones).



### Test Harness

## Puzzle 1 - ones

Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_9_0.png)
    


## Puzzle 2 - sum

Compute [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) - the sum of a vector.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_12_0.png)
    


## Puzzle 3 - outer

Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_15_0.png)
    


## Puzzle 4 - diag

Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_18_0.png)
    


## Puzzle 5 - eye

Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_21_0.png)
    


## Puzzle 6 - triu

Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_24_0.png)
    


## Puzzle 7 - cumsum

Compute [cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) - the cumulative sum.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_27_0.png)
    


## Puzzle 8 - diff

Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_30_0.png)
    


## Puzzle 9 - vstack

Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_33_0.png)
    


## Puzzle 10 - roll

Compute [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_36_0.png)
    


## Puzzle 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_39_0.png)
    


## Puzzle 12 - compress


Compute [compress](https://numpy.org/doc/stable/reference/generated/numpy.compress.html) - keep only masked entries (left-aligned).


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_42_0.png)
    


## Puzzle 13 - pad_to


Compute pad_to - eliminate or add 0s to change size of vector.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_45_0.png)
    


## Puzzle 14 - sequence_mask


Compute [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_48_0.png)
    


## Puzzle 15 - bincount

Compute [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_51_0.png)
    


## Puzzle 16 - scatter_add

Compute [scatter_add](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) - add together values that link to the same location.


    
![png](Tensor%20Puzzlers_files/Tensor%20Puzzlers_54_0.png)
    



# Speed Run Mode!

What is the smallest you can make each of these?

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

