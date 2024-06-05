# Boolean network generator for control and reduction

Automatically generates examples of Boolean networks from properties
on the controllability of their synchronous, aysnchronous, and general asynchronous dynamics
before and after reduction by variable elimination.

The program will exhaust all the possible Boolean networks of given dimension
until a solution is found, and can perform minimization.
An UNSAT answer indicates that no Boolean network of given dimension and
properties exist (modulo bugs).
It is generally not tractable above 6 dimensions.

⚠️ This code has been designed for generating counterexamples of control
relationships between a Boolean network and its reduction by variable
elimination studied in ["Phenotype control and elimination of variables in Boolean networks" by Tonello and Paulevé](https://arxiv.org/abs/2406.02304).\
**The structure, generality, and user-friendliness of the program is at a very preliminary stage.**
Yet, we believed it may give inspiration for generating (counter)examples for other works.

## Requirements

- `make`
- Python (at least 3.10)
- [clingo](https://potassco.org/clingo)
- `dot`

## Usage

1. Add desired specification at the end of `main.py`
2. Execute `python gen_asp_ce.py {Psize} {n} | clingo`, or\
   `make PSIZE={Psiez} gence{n}` (default to `{Psize}=1`)
   where `{n}` is the dimension, and `{Psize}` the number of components that
   must be controlled to 0.

   When using `make`, Boolean functions, influence graphs, and dynamics are
   extracted from the generated solution.
