# Yarrow Diagrams

[![Documentation Status](https://readthedocs.org/projects/yarrow-diagrams/badge/?version=latest)](https://yarrow-diagrams.readthedocs.io/en/latest/?badge=latest)

**⚠️ yarrow is still early in development ⚠️**: it's missing some features, and not everything is documented. Consider yourself warned!

[yarrow](https://yarrow.id) is a Python library implementing the datastructures
and algorithms for string diagrams described in the paper
[Data-Parallel Algorithms for String Diagrams][yarrow-paper].

# What is yarrow?

For a programmer's overview of what yarrow is for, see
[the documentation](https://yarrow-diagrams.readthedocs.io/).
In short, the main datastructure of [yarrow][yarrow] is the
[*Diagram*][diagram-docs], which can be thought of a generalisation of syntax
*trees* to syntax *graphs*.

[diagram-docs]: https://yarrow-diagrams.readthedocs.io/en/latest/_autosummary/yarrow.diagram.html

Here's an example of a string diagram depicted graphically:

<p align="center">
    <img src="https://yarrow.id/img/mean-background.png" style="height: 5rem;">
<p>

You could think of this as representing the *syntax* of the following python
program:

```py
def compute_mean(numbers: List[int]):
    count, sum = summary(numbers)
    return divide(sum, count)
```

String diagrams are different from directed graphs in two important ways:

- The boxes (operations) in a string diagram have multiple ordered inputs and outputs in the same way
  a python function has ordered *input arguments*.
- The diagram *itself* has inputs and outputs depicted as "dangling" wires on
  the left and right

Other examples of structures which can be represented by string diagrams are
*electronic circuits*, *quantum circuits*, *neural networks*, and many more.
For a more formal introduction to string diagrams aimed at computer scientists,
try [this recent paper](https://arxiv.org/abs/2305.08768).

# Installation

    pip install yarrow-diagrams

# Running tests

Install test dependencies

    pip install hypothesis

Run test with your test runner, e.g.

    pytest

# Yarrow as a Reference Implementation

Yarrow is intended as a *reference implementation* for the
[paper][yarrow-paper].
It has the following goals:

* Fast, data-parallel, and [runs on the GPU](./GPU_ACCELERATION.md)
* Minimal primitives/dependencies required
* Simple to implement correctly

The aim is to serve as a general-purpose datastructure in many languages
by making it simple to implement fast algorithms while relying
on few external dependencies.

To port yarrow to your language, the only things needed are:

* Arrays and simple array subroutines (see the [numpy backend](https://yarrow-diagrams.readthedocs.io/en/latest/_autosummary/_autosummary/yarrow.array.numpy.html#module-yarrow.array.numpy) or Section 2.2.2 of the [paper][yarrow-paper].)
* An algorithm for computing [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory)#Algorithms).

## Porting Yarrow

If you want to port Yarrow to a different language and you would like help,
please reach out. Here are some places you could do that:

- [discord](https://discord.gg/GWhkmQgMn7)
- `##yarrow` on `irc.libera.chat`
- email [the author](https://statusfailed.com/about.html)

[yarrow-paper]: https://arxiv.org/abs/2305.01041
[yarrow]: https://yarrow.id
