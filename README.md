## AllocVec

A vector where each element pushed is guaranteed to keep
its index until it is removed.

Internally, a normal `Vec` is used along with a linked list
for fast allocation and de-allocation.

## Example
```rust
use allocvec::AllocVec;

let mut vec = AllocVec::new();
let idx1 = vec.allocate(4);
let idx2 = vec.allocate(8);
let idx3 = vec.allocate(15);

vec.deallocate(idx1);
vec.deallocate(idx3);

assert_eq!(Some(8), vec.get(idx2));
```

## Speed

Though I haven't benchmarked the efficiency of the implementation,
**allocation**, **deallocation** and **indexing** are **O(1)**
operations, while calculating the length of the vector is **O(n)**.

Iterators need to filter unallocated and empty slots in the vector, so that
will most likely be bad for branch prediction.