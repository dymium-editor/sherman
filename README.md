# Sherman — a truly monstrous tree type

It turns out that in the production of an editor, sometimes specialized data structures are
used.  Sometimes data structures are similar enough that they can be united into a single
abstract type, with parameterizations to suit the original needs. This crate primarily exports
a single type -- `RleTree` -- which represents the collection of all features

### Notable features

* Values are retrieved by global index
* `RleTree` is named such for its run-length encoding -- individual entries in the tree
    represent a uniform range of indexes
* Efficient "shift" operations -- the details of the run-length encoding allow new ranges to be
    inserted in the middle, shifting everything after them, in 𝓞(log n) time
* Node references -- the position and values of a prior insertion can be fetched in 𝓞(log n)
    time, with relatively little overhead
* Lock-free concurrent {copy,clone}-on-write -- `RleTree`s can be shared across threads, with
    concurrent writes copying or cloning only the path down to the changed node(s).

And of course, all of these features are zero-cost when not in use: the tree is constructed in
such a way so that only the instances that actually *do* use these extra feature (like node
references or concurrent COW) have to pay the cost of them. And the cost of each individual
feature has been minimized as much as is possible.

For a more detailed explanation on how we're able to pull this off,
[architecture.md](architecture.md) gives a high-level overview of all the moving pieces to this
crate.

### Naming

This library is named after [General Sherman], a tree in Sequoia National Park that is the
current largest tree Earth by volume.

[General Sherman]: https://en.wikipedia.org/wiki/General_Sherman_(tree)
