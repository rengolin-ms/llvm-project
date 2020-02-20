Knossos MLIR libraries
======================

This tree aims to implement two libraries: a Knossos file reader that emits MLIR,
and an API providing introspection into an MLIR graph for heuristics search.

MLIR back-end
-------------

The MLIR back-end parsers Knossos IR and emits MLIR using as much as possible of
the MLIR core/affine/linalg dialects.

This back-end prints out an MLIR graph (text, binary) when requested, or it
feeds into the introspection API library for further processing.

The library should also be able to import MLIR graphs, for round-trip testing
and interacting with other MLIR tools.

**Partial implementation. See TODO below.**

MLIR Heuristics Introspection Engine
------------------------------------

Provides a stable way to query MLIR graphs for the purpose of heuristics
searches. We may need a Knossos-specific dialect.

If we do, we'll need to be able to annotate generic MLIR graph with Knossos
dialect, as well as read Knossos IR into MLIR into that dialect.

**Not implemented yet.**

Testing
-------

```bash
$ apt install clang ninja-build cmake lld

$ git clone git@github.com:rengolin-ms/llvm-project.git

$ cd llvm-project && git co ksc-mlir

$ mkdir build && cd build

$ cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON

$ ninja ksc-test && ./bin/ksc-test -vvv
```

If no error is reported (all "OK"), all is good.

**Tested with:**
 * Compilers: Clang 6 and 9, GCC 7.4 and 9.2
 * Linkers: GNU and LLD
 * OS: Ubuntu 18 and 19

Should work with any compiler, linker and OS LLVM works with.

Implementation Details
----------------------

The Lexer/Tokeniser creates two different kinds of tokens: values and non-values.
This is consistent with existing continuation compilers.

The expected sequence of value and non-value tokens and the token's value will
determine which kind of AST node we're parsing.

The Parser reads the Token tree and creates the AST, which is then used by the
MLIR back end to generate a module. The module can be optimised and LLVM IR
generated from it.

In the future, the heuristics engine will use the MLIR module to optimise the
graph, and there will be the option of either running the result in a JIT or
lowering to specific hardware.

**Back-end TODO:**
 * Add support for local variables (context) & finish Knossos IR support.
 * Run through multiple examples (../test), add FileCheck validation.
 * Allow MLIR round-trip, add tests for that.

**Engine TODO:**
 * Cost Model: liveness analysis, target-specific hooks, PGO, etc.
 * Navigation: find users/ops, dominance analysis, control flow, patterns, etc.
 * Rewrite dialect: known transformations representation, heuristics costs.
 * Heuristics: query/add/change heuristics values based on cost and predictions.
 * Traversal: allows sub-trees to be selected, pruned, cloned, modified, etc.
 * Lowering: either to a JIT execution engine or to specific hardware languages.
