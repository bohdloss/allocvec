#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::mem::replace;
use core::ops::{Index, IndexMut};

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
    /// * Some if the element is allocated at the index
    /// * None otherwise
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner
            .get(index)
            .and_then(|x| match x {
                AllocState::Unallocated(_) => None,
                AllocState::Allocated(elem) => Some(elem)
            })
    }

    /// Returns a mutable reference to an element at the given position
    ///
    /// # Arguments
    /// * `index` - The index of the element
    ///
    /// # Returns
    /// * Some if the element is allocated at the index
    /// * None otherwise
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner
            .get_mut(index)
            .and_then(|x| match x {
                AllocState::Unallocated(_) => None,
                AllocState::Allocated(elem) => Some(elem)
            })
    }

    /// Constructs a new vector containing all the elements of this
    /// `AllocVec` that are allocated
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.inner
            .into_iter()
            .map(|x| match x {
                AllocState::Unallocated(_) => None,
                AllocState::Allocated(elem) => Some(elem),
            })
            .flatten()
            .collect()
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
    pub fn alloc(&mut self, item: T) -> usize {
        self.alloc_cyclic(move |_| item)
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
    pub fn alloc_cyclic<F>(&mut self, f: F) -> usize
    where F: FnOnce(usize) -> T
    {
        let free_slot = self.inner.get(self.first).and_then(|x| {
            match x {
                AllocState::Unallocated(next) => Some(replace(&mut self.first, *next)),
                AllocState::Allocated(_) => None
            }
        });

        match free_slot {
            Some(index) => {
                self.inner.insert(index, AllocState::Allocated(f(index)));
                index
            },
            None => {
                let index = self.inner.len();
                self.first = index + 1;
                self.inner.push(AllocState::Allocated(f(index)));
                index
            }
        }
    }

    /// Deallocates the space at the given index if it is allocated.
    ///
    /// # Arguments
    /// * `index` - The index of the space to deallocate
    ///
    /// # Returns
    /// * `Some` if the space at the given index was allocated
    /// * `None` otherwise
    #[inline]
    pub fn dealloc(&mut self, index: usize) -> Option<T> {
        self.inner.get_mut(index).and_then(|state| {
            match state {
                AllocState::Unallocated(_) => None,
                AllocState::Allocated(_) => {
                    let old_first = replace(&mut self.first, index);
                    let old_state = replace(state, AllocState::Unallocated(old_first));
                    match old_state {
                        AllocState::Unallocated(_) => unreachable!(),
                        AllocState::Allocated(old) => Some(old)
                    }
                }
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
    /// * `Ok<T>` if the space was allocated, where `T` is the previous value. After this method
    /// returns `Ok` the space at `index` is to be considered populated with `item`.
    /// * `Err` containing `item` if the space was unallocated.
    #[inline]
    pub fn realloc(&mut self, index: usize, item: T) -> Result<T, T> {
        match self.inner.get_mut(index) {
            Some(state) => {
                match state {
                    AllocState::Unallocated(_) => Err(item),
                    AllocState::Allocated(old) => {
                        let old = replace(old, item);
                        Ok(old)
                    }
                }
            },
            None => Err(item)
        }
    }

    /// # Arguments
    /// * `index` - The index of the space to check
    ///
    /// # Returns
    /// * Whether an element is allocated at the given index
    #[inline]
    pub fn is_allocated(&self, index: usize) -> bool {
        self.inner
            .get(index)
            .is_some_and(|x| match x {
                AllocState::Unallocated(_) => false,
                AllocState::Allocated(_) => true
            })
    }

    /// # Returns
    /// * The total number of allocated elements
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.iter().filter(|x| match x {
            AllocState::Unallocated(_) => false,
            AllocState::Allocated(_) => true
        }).count()
    }

    /// # Returns
    /// * True if there are no allocated elements
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
            .filter(|(_, x)| if let AllocState::Allocated(_) = x { true } else { false })
            .map(|(i, x)| if let AllocState::Allocated(x) = x { (i, x) } else { unreachable!() })
    }

    /// Returns a mutable iterator over the populated spaces that acts in a way similar to
    /// `.iter_mut().enumerate()` except the index actually represents an allocation
    #[inline]
    pub fn enumerate_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> + '_ {
        self.inner.iter_mut()
            .enumerate()
            .filter(|(_, x)| if let AllocState::Allocated(_) = x { true } else { false })
            .map(|(i, x)| if let AllocState::Allocated(x) = x { (i, x) } else { unreachable!() })
    }

    /// Returns an immutable iterator over the populated spaces
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.inner.iter()
            .filter(|x| if let AllocState::Allocated(_) = x { true } else { false })
            .map(|x| if let AllocState::Allocated(x) = x { x } else { unreachable!() })
    }

    /// Returns a mutable iterator over the populated spaces
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.inner.iter_mut()
            .filter(|x| if let AllocState::Allocated(_) = x { true } else { false })
            .map(|x| if let AllocState::Allocated(x) = x { x } else { unreachable!() })
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
            if let AllocState::Allocated(value_) = item {
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
            AllocState::Allocated(ref val) => val,
            _ => panic!("Tried to index unallocated value")
        }
    }
}

impl <T> IndexMut<usize> for AllocVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self.inner[index] {
            AllocState::Allocated(ref mut val) => val,
            _ => panic!("Tried to index unallocated value")
        }
    }
}

impl<T> From<Vec<T>> for AllocVec<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        let mut inner = Vec::with_capacity(value.len());
        for item in value {
            inner.push(AllocState::Allocated(item));
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
    Allocated(T)
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
    let (idx1, idx2, idx3) = (vec.alloc(24), vec.alloc(98), vec.alloc(12));

    {
        let (get1, get2, get3) = (vec.get(idx1), vec.get(idx2), vec.get(idx3));
        assert!(get1.is_some());
        assert!(get2.is_some());
        assert!(get3.is_some());
    }

    assert_eq!(vec[idx1], 24);
    assert_eq!(vec[idx2], 98);
    assert_eq!(vec[idx3], 12);

    vec.dealloc(idx2);
    assert!(vec.get(idx1).is_some());
    assert!(vec.get(idx2).is_none());
    assert!(vec.get(idx3).is_some());

    vec.dealloc(idx1);
    assert!(vec.get(idx1).is_none());
    assert!(vec.get(idx2).is_none());
    assert!(vec.get(idx3).is_some());

    vec.dealloc(idx3);
    assert!(vec.get(idx1).is_none());
    assert!(vec.get(idx2).is_none());
    assert!(vec.get(idx3).is_none());
}

#[test]
fn alloc_vec_reallocation() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let (idx1, mut idx2, idx3) = (vec.alloc(456), vec.alloc(10), vec.alloc(14));

    assert_eq!(vec[idx1], 456);
    assert_eq!(vec[idx2], 10);
    assert_eq!(vec[idx3], 14);

    vec.dealloc(idx2);
    idx2 = vec.alloc(145);
    assert_eq!(vec[idx2], 145);
}

#[test]
fn alloc_vec_mut() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let idx = vec.alloc(15);

    assert_eq!(vec[idx], 15);
    vec[idx] = 20;
    assert_eq!(vec[idx], 20);
}

#[test]
fn alloc_vec_iter() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let idx1 = vec.alloc(15);
    let idx2 = vec.alloc(90);
    let idx3 = vec.alloc(75);
    let idx4 = vec.alloc(42);

    vec.dealloc(idx2);
    let _ = vec.realloc(idx3, 7);

    let collect: Vec<i32> = vec.iter().map(|x| *x).collect();
    assert_eq!(collect[0], 15);
    assert_eq!(collect[1], 7);
    assert_eq!(collect[2], 42);

    let collect: Vec<(usize, i32)> = vec.enumerate().map(|(i, x)| (i, *x)).collect();
    assert_eq!(collect[0], (idx1, 15));
    assert_eq!(collect[1], (idx3, 7));
    assert_eq!(collect[2], (idx4, 42));
}

#[test]
fn alloc_vec_getters() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let idx1 = vec.alloc(15);
    let idx2 = vec.alloc(90);
    let idx3 = vec.alloc(75);
    let idx4 = vec.alloc(42);

    vec.dealloc(idx2);
    let _ = vec.realloc(idx3, 8);

    assert!(vec.is_allocated(idx1));
    assert!(!vec.is_allocated(idx2));
    assert!(vec.is_allocated(idx3));
    assert!(vec.is_allocated(idx4));
}

#[test]
fn alloc_vec_cyclic() {
    let mut vec: AllocVec<i32> = AllocVec::new();
    let mut closure_idx = None;
    let idx = vec.alloc_cyclic(|idx| {
        closure_idx = Some(idx);
        42
    });

    assert_eq!(idx, closure_idx.unwrap());
}