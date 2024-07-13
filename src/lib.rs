#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::mem::replace;
use core::ops::{Index, IndexMut};
use AllocState::*;

/// Simple wrapper for Vec that handles allocating / deallocating
/// slots inside the vector, optimized for frequent indexing and
/// frequent allocations / de-allocations.
///
/// The index obtained through allocation is guaranteed to remain the same
/// and won't be reused until deallocated
#[derive(Clone)]
pub struct AllocVec<T> {
    first: usize,
    inner: Vec<AllocState<T>>
}

impl<T> AllocVec<T> {
    /// Creates a new `AllocVec` backed by an empty `Vec`
    #[inline]
    pub const fn new() -> Self {
        Self {
            first: 0,
            inner: Vec::new()
        }
    }

    /// Creates a new `AllocVec` backed by a `Vec` with the given capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            first: 0,
            inner: Vec::with_capacity(capacity)
        }
    }

    /// Returns an immutable reference to an element at the given position
    ///
    /// # Arguments
    /// * `index` - The index of the element
    ///
    /// # Returns
    /// * Some if the element is present, allocated and not reserved
    /// * None otherwise
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index).and_then(|x| x.as_populated())
    }

    /// Returns a mutable reference to an element at the given position
    ///
    /// # Arguments
    /// * `index` - The index of the element
    ///
    /// # Returns
    /// * Some if the element is present, allocated and not reserved
    /// * None otherwise
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut(index).and_then(|x| x.as_populated_mut())
    }

    /// Constructs a new vector containing all the elements of this
    /// `AllocVec` that are allocated and not reserved
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        let mut vec = Vec::new();
        for item in self.inner {
            if let Populated(item) = item {
                vec.push(item);
            }
        }
        vec
    }

    /// Allocates a space and populates it with `item`. No guarantees are made about
    /// the value of the returned index, except that it will remain valid
    /// until deallocated.
    ///
    /// # Arguments
    /// * `item` - The item to allocate
    ///
    /// # Returns
    /// * The index of the newly allocated item
    #[inline]
    pub fn allocate(&mut self, item: T) -> usize {
        self.allocate_cyclic(move |_| item)
    }

    /// Allocates a space and populates it with an item given by the closure `f`,
    /// which will receive the allocation index as a parameter. Useful for items that
    /// need to contain information about the index they are allocated in
    ///
    /// # Arguments
    /// * `f` - A closure that will receive the index and return the item to populate with
    ///
    /// # Returns
    /// * The index of the newly allocated space, equal to the index the closure sees
    #[inline]
    pub fn allocate_cyclic<F>(&mut self, f: F) -> usize
    where F: FnOnce(usize) -> T
    {

        let free_slot = self.inner.get(self.first).and_then(|x| {
            x.as_unallocated().map(|next| replace(&mut self.first, next))
        });

        match free_slot {
            Some(index) => {
                self.inner.insert(index, Populated(f(index)));
                index
            },
            None => {
                let index = self.inner.len();
                self.first = index + 1;
                self.inner.push(Populated(f(index)));
                index
            }
        }
    }

    /// Reserves a space. No guarantees are made about
    /// the value of the returned index, except that it will remain valid
    /// until deallocated.
    ///
    /// # Returns
    /// * The index of the newly reserved space
    #[inline]
    pub fn reserve(&mut self) -> usize {
        let free_slot = self.inner.get(self.first).and_then(|x| {
            x.as_unallocated().map(|next| replace(&mut self.first, next))
        });

        match free_slot {
            Some(index) => {
                self.inner.insert(index, Reserved);
                index
            },
            None => {
                let index = self.inner.len();
                self.first = index + 1;
                self.inner.push(Reserved);
                index
            }
        }
    }

    /// Deallocates the space at the given index if it is populated or reserved
    ///
    /// # Arguments
    /// * `index` - The index of the space to deallocate
    ///
    /// # Returns
    /// * `Some` if the space at the given index was populated
    /// * `None` otherwise
    #[inline]
    pub fn deallocate(&mut self, index: usize) -> Option<T> {
        self.inner.get_mut(index).and_then(|state| {
            if state.is_unallocated() {
                None
            } else {
                let old_first = self.first;
                self.first = index;
                let old_state = replace(state, Unallocated(old_first));
                old_state.into_populated()
            }
        })
    }

    /// Takes an element out of the space at the given index, possibly downgrading it
    /// from populated to just reserved. If this method returns `Some`, then the space
    /// at the given index is to be considered reserved. If `None` is returned instead,
    /// then it might have already been reserved or even unallocated
    ///
    /// # Arguments
    /// * `index` - The index of the space whose item is to take
    ///
    /// # Returns
    /// * `Some` if the element was populated
    /// * `None` otherwise
    #[inline]
    pub fn take(&mut self, index: usize) -> Option<T> {
        self.inner.get_mut(index).and_then(|state| {
            if state.is_unallocated() {
                None
            } else {
                let old = replace(state, Reserved);
                old.into_populated()
            }
        })
    }

    /// Replaces the element in the space at `index`, with `item`.
    ///
    /// # Arguments
    /// * `index` - The index of the space whose item to replace
    /// * `item` - The item to replace with
    ///
    /// # Returns
    /// * `Ok` if the space is populated or reserved, along with an `Option`
    /// that will be `Some` in case the space was populated, containing
    /// the previous value, `None` if the space was reserved. After this method
    /// returns `Ok` the space at `index` is to be considered populated with `item`.
    /// * `Err` containing `item` if the space is unallocated.
    #[inline]
    pub fn replace(&mut self, index: usize, item: T) -> Result<Option<T>, T> {
        match self.inner.get_mut(index) {
            Some(state) => {
                if state.is_unallocated() {
                    Err(item)
                } else {
                    let old = replace(state, Populated(item));
                    Ok(old.into_populated())
                }
            },
            None => Err(item)
        }
    }

    /// # Arguments
    /// * `index` - The index of the space to check
    ///
    /// # Returns
    /// * Whether the space at the given index is populated or reserved
    #[inline]
    pub fn is_allocated(&self, index: usize) -> bool {
        self.inner.get(index).is_some_and(|x| !x.is_unallocated())
    }

    /// # Arguments
    /// * `index` - The index of the space to check
    ///
    /// # Returns
    /// * Whether the space at the given index is populated and not reserved
    #[inline]
    pub fn is_populated(&self, index: usize) -> bool {
        self.inner.get(index).is_some_and(|x| x.is_populated())
    }

    /// # Arguments
    /// * `index` - The index of the space to check
    ///
    /// # Returns
    /// * Whether the space at the given index is reserved and not populated
    #[inline]
    pub fn is_reserved(&self, index: usize) -> bool {
        self.inner.get(index).is_some_and(|x| x.is_reserved())
    }

    /// # Returns
    /// * The total number of elements that are either reserved or populated
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.iter().filter(|x| !x.is_unallocated()).count()
    }

    /// # Returns
    /// * Ture if there are no reserved nor populated elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Returns
    /// * The length of the underlying `Vec`
    #[inline]
    pub fn vec_len(&self) -> usize {
        self.inner.len()
    }

    /// # Returns
    /// * The capacity of the underlying `Vec`
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns an immutable iterator over the populated spaces that acts in a way similar to
    /// `.iter().enumerate()` except the index actually represents an allocation
    #[inline]
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        self.inner.iter()
            .enumerate()
            .filter(|(_, x)| if let Populated(_) = x { true } else { false })
            .map(|(i, x)| if let Populated(x) = x { (i, x) } else { unreachable!() })
    }

    /// Returns a mutable iterator over the populated spaces that acts in a way similar to
    /// `.iter().enumerate()` except the index actually represents an allocation
    #[inline]
    pub fn enumerate_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> + '_ {
        self.inner.iter_mut()
            .enumerate()
            .filter(|(_, x)| if let Populated(_) = x { true } else { false })
            .map(|(i, x)| if let Populated(x) = x { (i, x) } else { unreachable!() })
    }

    /// Returns an immutable iterator over the populated spaces
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.inner.iter()
            .filter(|x| if let Populated(_) = x { true } else { false })
            .map(|x| if let Populated(x) = x { x } else { unreachable!() })
    }

    /// Returns a mutable iterator over the populated spaces
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.inner.iter_mut()
            .filter(|x| if let Populated(_) = x { true } else { false })
            .map(|x| if let Populated(x) = x { x } else { unreachable!() })
    }
}

impl<T: Eq> AllocVec<T> {
    /// Iterates through the vector searching for a slot that's populated with
    /// an item that is equal to the provided `value`.
    ///
    /// Requires `T` to implement [`Eq`]
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        for item in self.inner.iter() {
            if let Populated(value_) = item {
                if value_ == value {
                    return true;
                }
            }
        }
        false
    }
}

impl<T> Index<usize> for AllocVec<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match self.inner[index] {
            Populated(ref val) => val,
            _ => panic!("Tried to index unallocated value")
        }
    }
}

impl <T> IndexMut<usize> for AllocVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.inner[index] {
            Populated(ref mut val) => val,
            _ => panic!("Tried to index unallocated value")
        }
    }
}

impl<T> From<Vec<T>> for AllocVec<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        let mut inner = Vec::with_capacity(value.len());
        for item in value {
            inner.push(Populated(item));
        }
        Self {
            first: 0,
            inner
        }
    }
}

#[derive(Clone)]
enum AllocState<T> {
    Unallocated(usize),
    Reserved,
    Populated(T)
}

impl<T> AllocState<T> {
    fn is_unallocated(&self) -> bool {
        match self {
            Unallocated(_) => true,
            _ => false
        }
    }

    fn as_unallocated(&self) -> Option<usize> {
        match self {
            Unallocated(x) => Some(*x),
            _ => None
        }
    }

    fn is_reserved(&self) -> bool {
        match self {
            Reserved => true,
            _ => false
        }
    }

    fn is_populated(&self) -> bool {
        match self {
            Populated(_) => true,
            _ => false
        }
    }

    fn as_populated(&self) -> Option<&T> {
        match self {
            Populated(x) => Some(x),
            _ => None
        }
    }

    fn as_populated_mut(&mut self) -> Option<&mut T> {
        match self {
            Populated(x) => Some(x),
            _ => None
        }
    }

    fn into_populated(self) -> Option<T> {
        match self {
            Populated(x) => Some(x),
            _ => None
        }
    }
}

#[test]
fn alloc_vec_new() {
    let vec: AllocVec<i32> = AllocVec::new();
    assert_eq!(vec.vec_len(), 0);
    assert_eq!(vec.capacity(), 0);
}

#[test]
fn alloc_vec_with_capacity() {
    let vec: AllocVec<i32> = AllocVec::with_capacity(15);
    assert_eq!(vec.vec_len(), 0);
    assert!(vec.capacity() >= 15);
}

#[test]
fn alloc_vec_alloc_access_dealloc() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let (idx1, idx2, idx3) = (vec.allocate(24), vec.allocate(98), vec.allocate(12));

    {
        let (get1, get2, get3) = (vec.get(idx1), vec.get(idx2), vec.get(idx3));
        assert!(get1.is_some());
        assert!(get2.is_some());
        assert!(get3.is_some());
    }

    assert_eq!(vec[idx1], 24);
    assert_eq!(vec[idx2], 98);
    assert_eq!(vec[idx3], 12);

    vec.deallocate(idx2);
    assert!(vec.get(idx1).is_some());
    assert!(vec.get(idx2).is_none());
    assert!(vec.get(idx3).is_some());

    vec.deallocate(idx1);
    assert!(vec.get(idx1).is_none());
    assert!(vec.get(idx2).is_none());
    assert!(vec.get(idx3).is_some());

    vec.deallocate(idx3);
    assert!(vec.get(idx1).is_none());
    assert!(vec.get(idx2).is_none());
    assert!(vec.get(idx3).is_none());
}

#[test]
fn alloc_vec_reallocation() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let (idx1, mut idx2, idx3) = (vec.allocate(456), vec.allocate(10), vec.allocate(14));

    assert_eq!(vec[idx1], 456);
    assert_eq!(vec[idx2], 10);
    assert_eq!(vec[idx3], 14);

    vec.deallocate(idx2);
    idx2 = vec.allocate(145);
    assert_eq!(vec[idx2], 145);
}

#[test]
fn alloc_vec_mut() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let idx = vec.allocate(15);

    assert_eq!(vec[idx], 15);
    vec[idx] = 20;
    assert_eq!(vec[idx], 20);
}

#[test]
fn alloc_vec_iter() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let idx1 = vec.allocate(15);
    let idx2 = vec.allocate(90);
    let idx3 = vec.allocate(75);
    let idx4 = vec.allocate(42);

    vec.deallocate(idx2);
    vec.take(idx3);

    let collect: Vec<i32> = vec.iter().map(|x| *x).collect();
    assert_eq!(collect[0], 15);
    assert_eq!(collect[1], 42);

    let collect: Vec<(usize, i32)> = vec.enumerate().map(|(i, x)| (i, *x)).collect();
    assert_eq!(collect[0], (idx1, 15));
    assert_eq!(collect[1], (idx4, 42));
}

#[test]
fn alloc_vec_getters() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let idx1 = vec.allocate(15);
    let idx2 = vec.allocate(90);
    let idx3 = vec.allocate(75);
    let idx4 = vec.allocate(42);

    vec.deallocate(idx2);
    vec.take(idx3);

    assert!(vec.is_allocated(idx1));
    assert!(!vec.is_allocated(idx2));
    assert!(vec.is_allocated(idx3));
    assert!(vec.is_allocated(idx4));

    assert!(vec.is_populated(idx1));
    assert!(vec.is_reserved(idx3));
    assert!(vec.is_populated(idx4));
}

#[test]
fn alloc_vec_cyclic() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let mut closure_idx = None;
    let idx = vec.allocate_cyclic(|idx| {
        closure_idx = Some(idx);
        42
    });

    assert_eq!(idx, closure_idx.unwrap());
}