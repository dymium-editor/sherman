# Architecture

Internally, `RleTree` is a modified [AVL tree](https://en.wikipedia.org/wiki/AVL_tree).

**Table of Contents:**

- [`RleTree` nodes](#rletree-nodes)
- [`NodeHandle` and borrowing](#nodehandle-and-borrowing)
- [Fuzzing](#fuzzing)

## `RleTree` nodes

*See: `src/tree/node.rs`*

Each `RleTree` node stores:

* The value ("slice", `S`)
* The total size of the subtree rooted at the node
* The height of the subtree rooted at the node
* Child & parent pointers

The size of a node's value is determined by subtracing the left & right child subtree sizes from the
node's subtree size (producing just the middle part not present in either child).

Parent pointers also allow us to implement all tree operations with constant-memory loops intead of
recursion (with the small exception of iteration on COW-enabled trees, where parent pointers aren't
guaranteed to match any particular shallow copy).

Each node also stores:

* If COW is enabled...
  * An `AtomicU64` for the number of "strong" references (shallow clones) to the node
* If stable refs are enabled...
  * A `Cell<u64>` for the number of "weak" references (stable ref *or* redirect) to the node
  * A `Cell<BorrowState>` to prevent misbehaved index/value types from violating the 
  * A `Cell<Option<(redirect pointer)>>` for redirecting stable refs to a new node, if the value
    originally in this node was merged into another.

The `RleTreeConfig` parameter `P` uses a handful of associated types & helper traits in order to
only include these fields when the configuration requires it.

## `NodeHandle` and borrowing

*See: `src/tree/node.rs`, `src/tree/borrow.rs`*

On its own, there's a lot of complexity here: We're managing our own allocations, with subtle
ownership patterns — parent pointers, optional multi-threaded copy-on-write, optional
single-threaded "weak" references.

To manage that, we have a `NodeHandle` abstraction providing semantics very similar to `Box<Node>`,
`&Node`, and `&mut Node`, among some other variants (possibly-`Arc<Node>`, `rc::Weak<Node>`).

A key rule: We *never* hand out references to a `Node`; instead, we use `NodeHandle`s to produce
references to individual fields with the `borrow::Field` trait and macro-generated types +
implementations for `Node` fields. Otherwise, it gets too easy to violate the stacked borrows rules.
(And, this also means certain fields are enforced as *never* mutable, which allows reentrant access
to just those fields by weak handles, even when there's an ongoing mutation.)

The "owned" & "weak" variants of these handles implement destructors that ensure the values are
eventually dropped and the memory eventually deallocated. This helps make higher level operations
correct by default (e.g., ensuring we don't leak memory if we panic after temporarily removing a
node from the tree).

## Fuzzing

*See: `src/fuzz/fake.rs`, `src/fuzz/arbitrary_operation.rs`*

Our primary tool for ensuring correctness is fuzzing.

`RleTree` is, in some sense, a datastructure that could naively be implemented as just a `Vec` with
*O(n)* operations instead of *O(log n)*. So we have a `Fake` type that acts exactly as a simpler,
easier to validate version of the same interface.

Fuzzing is built on top of `Fake`, with the `ArbitraryOperation` type (e.g., get, insert, remove).

A sequence of operations implements [`Arbitrary`], where:

1. During initial construction of the operation sequence, we run the operation against a `Fake`
   (built up from the prior operations) and record the result
2. During execution, we run the operation against a real `RleTree` and check that it produces the
   same results as we got from the `Fake`

[`Arbitrary`]: https://docs.rs/arbitrary/latest/arbitrary/trait.Arbitrary.html

Additionally, the sequence of `ArbitraryOperation`s implements `Debug` such that, on failure,
[cargo-fuzz] prints a unit test reproducing the failure. Tests generated from prior failures are all
under `src/tree/tests/fuzz_*.rs`.

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
