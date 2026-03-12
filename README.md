# Sherman — a monstrous tree type

WIP. This is horror of a datastructure.

### Notable features

* Values are retrieved by global index
* `RleTree` is named such for its run-length encoding -- individual entries in the tree
    represent a uniform range of indexes
* Efficient "shift" operations -- the details of the run-length encoding allow new ranges to be
    inserted in the middle, shifting everything after them, in O(log n) time
* Node references -- the position and values of a prior insertion can be fetched in O(log n)
    time, with relatively little overhead
* Lock-free concurrent {copy,clone}-on-write -- `RleTree`s can be shared across threads, with
    concurrent writes copying or cloning only the path down to the changed node(s).

And of course, all of these features are zero-cost when not in use: the tree is constructed in
such a way so that only the instances that actually *do* use these extra feature (like node
references or concurrent COW) have to pay the cost of them.

### Naming

This library is named after [General Sherman], a tree in Sequoia National Park that is the
current largest tree Earth by volume.

[General Sherman]: https://en.wikipedia.org/wiki/General_Sherman_(tree)
