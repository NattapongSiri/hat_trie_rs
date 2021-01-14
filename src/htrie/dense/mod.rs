use super::*;

use num_traits::{bounds::Bounded, AsPrimitive, FromPrimitive};
use core::borrow::Borrow;

/// Find a element `K` that will split the given `bucket` in half at specified element index of key.
/// For example: if key is ["a", "b", "c"] and ["a", "c", "d"], given index is 1, the returned `K` will be
/// "c". If given idex is 2, the return `K` will be "d". This is because when index is 1, it look at second
/// element of both key thus it found "b" and "c". When index is 2, it look at third element which is
/// "c" and "d".
fn find_half_point<K, V>(bucket: &ArrayHash<twox_hash::XxHash64, <K as ToOwned>::Owned, V>, index: usize, start: usize, end: usize) -> <K as Index<usize>>::Output 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {
    // Find split point
    let mut count = vec![0; end - start + 1]; // Save some memory by allocate only necessary size
    let mut split_portion = 0;

    for (k, _) in bucket.iter() {
        let k = k.borrow();
        if k.len() > index {
            count[k[index].as_() - start] += 1; // Accumulate each individual key
            split_portion += 1; // Accumulate total number of entry
        }
    }

    split_portion /= 2; // Now we got split portion

    let mut split_point = 0;
    let mut first_portion = 0;
    let last_split_point = count.len() - 1;

    // It shall not be split beyond end point
    while first_portion < split_portion && split_point < last_split_point {
        first_portion += count[split_point];
        split_point += 1;
    }

    split_point += start;

    <K as Index<usize>>::Output::from_usize(split_point).unwrap()
}

/// Construct a hat-trie hybrid container by using unsafe raw pointer to skip
/// lookup time.
/// This is possible for "dense" type only as each key element can be converted into
/// usize to directly index into child node.
/// 
/// By convention, this is safe. It is because it use raw pointer only by internal
/// operation which when mutation occur, it occur under `&mut self` of outer type thus
/// make borrow checker enforce borrow rule as a whole on outer object.
#[derive(Clone, Debug)]
struct ChildsContainer<K, T, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      T: TrieNode<K, V>,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {
    /// Owned childs of a node. It need to be pre-allocated to prevent re-allocation which
    /// if happen, will invalidate all pointers that point to it.
    childs: Vec<NodeType<K, T, V>>,
    threshold: usize,
    _internal_pointer: Box<[*mut NodeType<K, T, V>]>
}

/// It should be safe to send this container between thread as it doesn't leak pointer out nor
/// point to anything outside it own value which guarded by borrow checker of Rust
unsafe impl<K, T, V> Send for ChildsContainer<K, T, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      T: TrieNode<K, V>,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {
}

/// Type have no use of any interior mutability nor any ref count type so it'd be safe
/// to sync
unsafe impl<K, T, V> Sync for ChildsContainer<K, T, V>
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      T: TrieNode<K, V>,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {}

impl<K, T, V> ChildsContainer<K, T, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      T: TrieNode<K, V>,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {
    /// Construct an empty childs container which initialize to Single "hybrid" node and
    /// all pointer in parent point to this single child.
    pub fn new(size: usize) -> Self {
        let mut childs = Vec::with_capacity(size);
        let start = <K as Index<usize>>::Output::min_value().as_();
        let end = <K as Index<usize>>::Output::max_value().as_();

        let bucket = NodeType::Hybrid((ArrayHashBuilder::default().build(), <K as Index<usize>>::Output::from_usize(start).unwrap()..=<K as Index<usize>>::Output::from_usize(end).unwrap()));
        childs.push(bucket);
        let ptrs = (0..size).map(|_| (&mut childs[0]) as *mut NodeType<K, T, V>).collect();
        ChildsContainer {
            childs,
            threshold: super::BURST_THRESHOLD,
            _internal_pointer: ptrs
        }
    }
    /// Construct an empty childs container with given specification.
    /// It is initialize to Single "hybrid" node and all pointer in parent point to this single child.
    /// 
    /// # Parameters
    /// - `size` - Number of slot this container going to provide. These slots are for redirecting index
    /// access to underlying childs which may be trie or `ArrayHash`
    /// - `threshold` - A maximum number of element before this container split it child. It must be greater thaan 0.
    /// - `init_bucket_size` - Initial size of `ArrayHash` in container node. It must be greater than 0.
    /// - `bucket_load_factor` - Number of elements per container before it scale.
    /// Each container will scale by multiple of this value.
    /// 
    /// # Return
    /// A container node with specified spec
    pub fn with_spec(size: usize, threshold: usize, init_bucket_size: usize, init_bucket_slots: usize, bucket_load_factor: usize) -> Self {
        debug_assert!(threshold > 0);
        debug_assert!(init_bucket_size > 0);
        let mut childs = Vec::with_capacity(size);
        let start = <K as Index<usize>>::Output::min_value().as_();
        let end = <K as Index<usize>>::Output::max_value().as_();
        let bucket = NodeType::Hybrid((ArrayHashBuilder::default()
                                                            .buckets_size(init_bucket_size)
                                                            .slot_size(init_bucket_slots)
                                                            .max_load_factor(bucket_load_factor)
                                                            .build(), 
                                            <K as Index<usize>>::Output::from_usize(start).unwrap()..=<K as Index<usize>>::Output::from_usize(end).unwrap()));
        childs.push(bucket);
        let ptrs = (0..size).map(|_| (&mut childs[0]) as *mut NodeType<K, T, V>).collect();
        ChildsContainer {
            childs,
            threshold,
            _internal_pointer: ptrs
        }
    }
    /// Attempt to split a child node at given key.
    /// 
    /// If the child node have element >= threshold, it will
    /// split the child according to hat-trie paper.
    /// 
    /// Since splitting child effect it parent pointer, we need to 
    /// perform it in this level.
    /// 
    /// # Return
    /// True if it was split. Otherwise, it return false.
    fn maybe_split(&mut self, key: <K as Index<usize>>::Output) -> bool {
        // This is a new trie which will be parent of new hybrid node yield from pure node split.
        let mut pure_trie = None;
        // This is a result of hybrid node split.
        let mut split_result = None;
        let threshold = self.threshold;
        match &mut self[key] {
            NodeType::Pure(bucket) => {
                if bucket.len() >= threshold {
                    // flag that current key need to be burst/split
                    pure_trie = Some(key);
                } else {
                    return false
                }
            },
            NodeType::Hybrid((bucket, range)) => {
                if bucket.len() >= threshold {
                    // Need to split
                    let start = range.start().as_();
                    let end = range.end().as_();

                    // Find split point
                    let split_point = find_half_point::<K, V>(bucket, 0, start, end);
                    // Now we got split point

                    // Make a new bucket to store half of old bucket
                    let new_bucket = bucket.split_by(|(k, _)| {
                        k.borrow()[0] >= split_point
                    });

                    *range = <K as Index<usize>>::Output::from_usize(start).unwrap()..=<K as Index<usize>>::Output::from_usize(split_point.as_() - 1).unwrap();

                    split_result = Some((new_bucket, [start, split_point.as_(), end]));
                } else {
                    return false
                }
            },
            // Other type doesn't have to be split
            _ => ()
        };

        if let Some(key) = pure_trie {
            // Post processing for Pure bucket
            let k = key.as_();

            // Need unsafe way to get access to short circuit scanning for an owned child.
            // This should be safe as the child is owned by itself in here.
            // It can also be done in completely safe code by scanning `self.childs` looking
            // for a child with a key having k as prefix but it going to be significantly slower
            // than de-reference the pointer.

            // Temporary take out child at K
            unsafe {
                let old_child = std::mem::take(&mut *self._internal_pointer[k]);
                
                // It is only possible to be "Pure" node type to reach here
                if let NodeType::Pure(bucket) = old_child {
                    // Consume bucket and make a trie out of it
                    let trie = NodeType::Trie(T::new_split(bucket, self.threshold));
                    // Place new child at K
                    *self._internal_pointer[k] = trie;
                }
            }
            return true
        } else if let Some((new_bucket, [start, split_point, end])) = split_result {
            // Post processing for Hybrid split.
            // 1. change existing bucket to pure if needed. 2. update parent pointers for new bucket
            if start == split_point - 1 {
                // The only case where we need to make node Pure
                let old = std::mem::take(&mut self[<K as Index<usize>>::Output::from_usize(start).unwrap()]);
                // The only possible type in here is Hybrid
                if let NodeType::Hybrid((table, _)) = old {
                    // Range can only be one here so we ignore it.

                    // Now replace the old Hybrid with Pure type
                    self[key] = NodeType::Pure(table);
                }
            }

            if split_point < end {
                self.childs.push(NodeType::Hybrid((new_bucket, <K as Index<usize>>::Output::from_usize(split_point).unwrap()..=<K as Index<usize>>::Output::from_usize(end).unwrap())));
            } else {
                // Only single possible parent
                self.childs.push(NodeType::Pure(new_bucket));
            }

            // Last element of self.childs hold a new bucket, we need to update all remain pointer
            let last = self.childs.len() - 1;
            
            // Update all remaining pointer to point to new half
            for ptr in self._internal_pointer[split_point..].iter_mut() {
                *ptr = (&mut self.childs[last]) as *mut NodeType<K, T, V>;
            }
            
            return true
        }
        false
    }
}

impl<K, T, V> core::ops::Index<<K as Index<usize>>::Output> for ChildsContainer<K, T, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      T: TrieNode<K, V>,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {
    type Output=NodeType<K, T, V>;

    fn index(&self, idx: <K as Index<usize>>::Output) -> &Self::Output {
        unsafe {
            &*self._internal_pointer[idx.as_()]
        }
    }
}

impl<K, T, V> core::ops::IndexMut<<K as Index<usize>>::Output> for ChildsContainer<K, T, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      T: TrieNode<K, V>,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {

    fn index_mut(&mut self, idx: <K as Index<usize>>::Output) -> &mut Self::Output {
        unsafe {
            &mut *self._internal_pointer[idx.as_()]
        }
    }
}

/// A type of Trie that implemented by using Vec index as encoded key.
/// 
/// Care should be taken before using this Trie.
/// 
/// The memory requirement for this Trie is equals to bits of key. For example, if
/// key is of type &[u8], each layer of Trie will need at least 2^8 * 4 bytes (1024 bytes) 
/// + all other meta such as minimum and maximum length of stored key, threshold for burst/split, etc.
/// If key is of type &[u32], each layer will need 2^32 * 4 bytes (16GB).
#[derive(Clone, Debug)]
pub struct DenseVecTrieNode<K, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {
    childs: ChildsContainer<K, Self, V>,
    max_len: usize,
    min_len: usize,
    threshold: usize,
    value: Option<V>,
}

impl<K, V> TrieNode<K, V> for DenseVecTrieNode<K, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {
    fn new_split(mut bucket: ArrayHash<twox_hash::XxHash64, <K as ToOwned>::Owned, V>, threshold: usize) -> Self {
        let start = <K as Index<usize>>::Output::min_value().as_();
        let end = <K as Index<usize>>::Output::max_value().as_();
        let mut old_size = bucket.len();
        let split_point = find_half_point::<K, V>(&bucket, 1, start, end);
        let mut node_value = None;
        let mut left = ArrayHashBuilder::default().build();
        let mut right = ArrayHashBuilder::default().build();
        let mut key_len = (usize::MAX, usize::MIN);

        for (key, value) in bucket.drain() {
            let key = key.borrow();
            let n = key.len();
            if n == 1 {
                key_len = (1, 1);
                // There's only single character in key.
                // It value must be present on new Trie node and remove itself from bucket.
                node_value = Some(value);
                old_size -= 1;
                continue;
            } else {
                if n > key_len.1 {
                    key_len.1 = n;
                }
                if n < key_len.0 {
                    key_len.0 = n;
                }
            }
            if key[1] >= split_point {
                let r = right.put(key[1..key.len()].to_owned(), value);
                debug_assert!(r.is_none());
            } else {
                let l = left.put(key[1..key.len()].to_owned(), value);
                debug_assert!(l.is_none());
            }
        }
        debug_assert_eq!(old_size, left.len() + right.len());

        // Construct a child container manually as we need to properly align each
        // K with correct side of bucket.
        // In future, if other struct also need this, we shall move this to ChildsContainer struct

        // We need to allocate enough space for childs mapping to prevent re-allocation which will invalidate
        // all pointers below
        let mut childs = Vec::with_capacity(end - start + 1);
        let split_point_usize = split_point.as_();
        
        let ptr = if split_point_usize == start {
            // Only one side available
            childs.push(NodeType::Hybrid((right, split_point..=<K as Index<usize>>::Output::max_value())));

            (start..=end).map(|_| {
                (&mut childs[0]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
            }).collect::<Vec<*mut NodeType<K, DenseVecTrieNode<K, V>, V>>>().into_boxed_slice()
        } else if split_point_usize == start + 1 {
            // Splitted into left and right. Left is pure. Right is hybrid
            childs.push(NodeType::Pure(left));
            childs.push(NodeType::Hybrid((right, split_point..=<K as Index<usize>>::Output::max_value())));

            std::iter::once((&mut childs[0]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>)
                        .chain((1..=end).map(|_| {
                            (&mut childs[1]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
                        })).collect::<Vec<*mut NodeType<K, DenseVecTrieNode<K, V>, V>>>().into_boxed_slice()
        } else if split_point_usize < end - 1 {
            // Splitted into left and right, both hybrid
            childs.push(NodeType::Hybrid((left, <K as Index<usize>>::Output::min_value()..=<K as Index<usize>>::Output::from_usize(split_point_usize - 1).unwrap())));
            childs.push(NodeType::Hybrid((right, split_point..=<K as Index<usize>>::Output::from_usize(end).unwrap())));

            let split_point_usize = split_point.as_();
            (start..=end).map(|key| {
                if key >= split_point_usize {
                    (&mut childs[1]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
                } else {
                    (&mut childs[0]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
                }
            }).collect::<Vec<*mut NodeType<K, DenseVecTrieNode<K, V>, V>>>().into_boxed_slice()
        } else {
            // Splitted into left and right. Left is hybrid. Right is pure
            childs.push(NodeType::Hybrid((left, <K as Index<usize>>::Output::min_value()..=<K as Index<usize>>::Output::from_usize(split_point_usize - 1).unwrap())));
            childs.push(NodeType::Pure(right));

            let split_point_usize = split_point.as_();
            (start..=end).map(|key| {
                if key >= split_point_usize {
                    (&mut childs[1]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
                } else {
                    (&mut childs[0]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
                }
            }).collect::<Vec<*mut NodeType<K, DenseVecTrieNode<K, V>, V>>>().into_boxed_slice()
        };

        DenseVecTrieNode {
            childs: ChildsContainer {
                childs,
                threshold,
                _internal_pointer: ptr
            },
            min_len: key_len.0,
            max_len: key_len.1,
            threshold,
            value: node_value,
        }
    }

    fn child<'a>(&'a self, key: &<K as Index<usize>>::Output) -> &'a NodeType<K, Self, V> {
        &self.childs[*key]
    }

    fn value(&self) -> Option<&V> {
        self.value.as_ref()
    }

    fn value_mut(&mut self) -> &mut Option<V> {
        &mut self.value
    }

    fn put(&mut self, key: &K, value: V) -> Option<V> {
        if key.len() == 0 {return None} // Cannot put empty key into trie

        if key.len() > self.max_len {
            self.max_len = key.len();
        }
        if key.len() < self.min_len {
            self.min_len = key.len();
        }

        let mut offset = 0; // Progress of key being process
        // let mut nodes = &mut self.childs; // Set root childs as nodes to be search for a key
        let mut parent = self;
        let mut nodes = &mut parent.childs; // Node context, all siblings are here
        nodes.maybe_split(key[0]); // Check if root node need to be split
        let mut node; // Current node being processed
        loop {
            // Dense Trie can use byte as index into each possible slot
            node = &mut nodes[key[offset]];

            match node {
                NodeType::None => {
                    // The byte is not part of trie. Add remain key and value to this trie here.
                    // Make a new container node to store value.
                    let mut bucket = ArrayHashBuilder::default().build();
                    bucket.put(key[offset..key.len()].to_owned(), value);
                    *node = NodeType::Pure(bucket);

                    return None
                }
                NodeType::Trie(t) => {
                    // For simple trie, we just add all childs back to node waiting to be process
                    parent = t;

                    offset += 1;

                    if offset >= key.len() {
                        // We exhausted the key, this trie node hold a value
                        return parent.value.replace(value)
                    }
                    nodes = &mut parent.childs;
                },
                NodeType::Pure(childs) => {
                    // For Pure type, it's `ArrayHash` that contains whole remaining key.
                    // We can simply leverage `put` method using remaining key and value.
                    let old = childs.put(key[offset..key.len()].to_owned(), value);
                    
                    if let Some((_, v)) = old {
                        return Some(v)
                    } else {
                        return None
                    }
                },
                NodeType::Hybrid(bucket) => {
                    // For Hybrid type, it has left and right and split point to determine
                    // which `ArrayHash` to put the remaining key and value.
                    let old = bucket.0.put(key[offset..key.len()].to_owned(), value);
                    if let Some((_, v)) = old {
                        return Some(v)
                    } else {
                        return None
                    }
                }
            }
        }
    }

    fn try_put<'a>(&'a mut self, key: &K, value: V) -> Option<&'a V> where K: 'a {
        if key.len() == 0 {return None} // Cannot put empty key into trie

        // It's safe to assume that if key len is new extreme, it's unique key.
        // Therefore, it should be able to put into this node.
        if key.len() > self.max_len {
            self.max_len = key.len();
        }
        if key.len() < self.min_len {
            self.min_len = key.len();
        }

        let mut offset = 0; // Progress of key being process
        let mut parent = self;
        let mut nodes = &mut parent.childs; // Set root childs as nodes to be search for a key
        let mut node; // Current node being processed
        loop {
            // Dense Trie can use byte as index into each possible slot
            node = &mut nodes[key[offset]];

            if offset < key.len() {
                // if there's some remain key to be process, we analyze child of current node
                match node {
                    NodeType::None => {
                        // The byte is not part of trie. Add remain key and value to this trie here.
                        // Make a new container node to store value.
                        let mut bucket = ArrayHashBuilder::default().build();
                        bucket.put(key[offset..key.len()].to_owned(), value);
                        *node = NodeType::Pure(bucket);

                        return None
                    },
                    NodeType::Trie(t) => {
                        // For simple trie, we just add all childs back to node waiting to be process
                        parent = t;
                        nodes = &mut parent.childs;
                    },
                    NodeType::Pure(childs) => {
                        // For Pure type, it's `ArrayHash` that contains whole remaining key.
                        // We can simply leverage `try_put` method using remaining key and value.
                        return childs.try_put(key[offset..key.len()].to_owned(), value)
                    },
                    NodeType::Hybrid(bucket) => {
                        // For Hybrid type, it has left and right and split point to determine
                        // which `ArrayHash` to try put the remaining key and value.
                        return bucket.0.try_put(key[offset..key.len()].to_owned(), value)
                    }
                }
            } else {
                match node {
                    NodeType::Trie(ref mut t) => {
                        // all bytes in key are consumed, return last child value
                        if t.value.is_none() {
                            t.value.replace(value);
                            return None
                        } else {
                            return t.value.as_ref()
                        }
                    },
                    _ => {
                        // shall never reach this block as either it is still trying to process trie
                        // or it is in a case where Pure/Hybrid container was met
                        return None
                    }
                }
            }
            offset += 1; // Move offset so next time key being use will be those that is not yet processed.
        }
    }

    #[inline(always)]
    fn max_key_len(&self) -> usize {
        self.max_len
    }

    #[inline(always)]
    fn min_key_len(&self) -> usize {
        self.min_len
    }
}

// impl<'a, K, V> super::StartsWith<'a, K, V> for DenseVecTrieNode<K, V>
// where K: AsPrimitive<usize> + Bounded + Copy + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
//       Box<[K]>: Clone + PartialEq,
//       V: 'a + Clone + core::fmt::Debug + Default {
//     type Iter=StartsWithIterator<'a, K, V>;
//     type IterItem=StartsWith<'a, K, V>;

//     fn starts_with(&'a self, key: &'a [K]) -> StartsWithIterator<'a, K, V> {
//         StartsWithIterator {
//             bucket: None,
//             cursors: vec![key[0].as_()],
//             level: 0,
//             nodes: vec![self],
//             prefixes: Vec::new(),
//             query: key
//         }
//     }
// }

impl<K, V> DenseVecTrieNode<K, V> 
where K: TrieKey + Hash + Index<usize> + Index<Range<usize>, Output=K> + PartialEq + ToOwned + ?Sized,
      V: Clone,
      <K as Index<usize>>::Output: AsPrimitive<usize> + Bounded + Clone + Debug + FromPrimitive + PartialEq + PartialOrd,
      <K as ToOwned>::Owned: Clone + Debug + Hash + PartialEq + for <'r> PartialEq<&'r K> {

    /// Create new empty [DenseVecTrieNode](struct.DenseVecTrieNode.html).
    /// 
    /// It will automatically expand into larger trie by append necesseary child type to
    /// maintain it fitness. This method return default configuration for the 
    /// [DenseVecTrieNode](struct.DenseVecTrieNode.html).
    pub fn new() -> DenseVecTrieNode<K, V> {
        DenseVecTrieNode {
            min_len: 0,
            max_len: 0,
                // root always has no value
            value: None,
            threshold: super::BURST_THRESHOLD,
            // Allocate Trie with length equals to Trie size. Since K is not `Clone`, we need to manually create all of it.
            childs: ChildsContainer::new(2usize.pow(std::mem::size_of::<<K as Index<usize>>::Output>() as u32 * 8u32))
        }
    }

    /// Create new empty [DenseVecTrieNode](struct.DenseVecTrieNode.html).
    /// 
    /// It will automatically expand into larger trie by append necesseary child type to
    /// maintain it fitness. This method take specifications which this trie should maintain in order
    /// to fit with expected performance.
    /// 
    /// # Parameters
    /// - `threshold` - The number of element in each leaf container which if exceed will cause node
    /// burst/split depending on type of node at that moment. It must be greater than 0.
    /// - `init_bucket_size` - The initial size of bucket. Bucket is a container used in leaf node.
    /// This number shall be large enough to have only few data in each slot in bucket but it shall
    /// also be small enough to fit into single page of memory for caching reason.
    /// This value must be greater than 0.
    /// - `init_bucket_slots` - The initial size of each slot which will store actual data. It will grow if
    /// there is a lot of hash collision.
    /// - `bucket_load_factor` - The number of element in bucket which if reached, will expand the bucket size
    /// by 2 times of current size.
    /// 
    /// # Return
    /// A trie instance of type [DenseVecTrieNode](struct.DenseVecTrieNode.html)
    pub fn with_spec(threshold: usize, init_bucket_size: usize, init_bucket_slots: usize, bucket_load_factor: usize) -> DenseVecTrieNode<K, V> {
        debug_assert!(threshold > 0);
        debug_assert!(init_bucket_size > 0);
        DenseVecTrieNode {
            min_len: 0,
            max_len: 0,
                // root always has no value
            value: None,
            threshold,
            // Allocate Trie with length equals to Trie size. Since K is not `Clone`, we need to manually create all of it.
            childs: ChildsContainer::with_spec(2usize.pow(std::mem::size_of::<<K as Index<usize>>::Output>() as u32 * 8u32), threshold, init_bucket_size, init_bucket_slots, bucket_load_factor)
        }
    }
}

// pub struct StartsWith<'a, K, V>
// where K: Copy + AsPrimitive<usize> + Bounded + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
//       V: Clone + core::fmt::Debug {
//     prefix: &'a [K],
//     key: &'a [K],
//     value: &'a V
// }

// impl<'a, K, V> super::StartsWithItem<K, V> for StartsWith<'a, K, V>
// where K: Copy + AsPrimitive<usize> + Bounded + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
//       V: Clone + core::fmt::Debug {
//     /// Retrieve a key in format of owned Vec that hold reference to each K in key in Trie.
//     fn key(&self) -> Vec<&K> {
//         self.prefix.iter().chain(self.key.iter()).collect()
//         // self.key.iter().map(|part| part.iter()).flatten().collect()
//         // self.key.as_slice()
//     }
//     /// Retrieve a borrowed value which shall reference to value stored in Trie
//     fn value(&self) -> &V {
//         self.value
//     }
// }

// pub struct StartsWithIterator<'a, K, V>
// where K: Copy + AsPrimitive<usize> + Bounded + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
//       Box<[K]>: Clone + PartialEq,
//       V: Clone + core::fmt::Debug + Default {
//     bucket: Option<ArrayHashIterator<'a, Box<[K]>, V>>,
//     cursors: Vec<usize>,
//     level: usize,
//     nodes: Vec<&'a DenseVecTrieNode<K, V>>,
//     prefixes: Vec<K>,
//     query: &'a [K],
// }

// impl<'a, K, V> Iterator for StartsWithIterator<'a, K, V>
// where K: Copy + AsPrimitive<usize> + Bounded + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
//       Box<[K]>: Clone + PartialEq,
//       V: Clone + core::fmt::Debug + Default {
//     type Item=StartsWith<'a, K, V>;

//     fn next(&mut self) -> Option<StartsWith<'a, K, V>> {
//         if let Some(ref mut it) = self.bucket {
//             if let Some((key, value)) = it.next() {
//                 // Construct a result container which has some prefix from Trie
//                 // and some other from bucket
//                 return Some(StartsWith {
//                     prefix: self.prefixes.as_slice(),
//                     key: &key[1..],
//                     value: value
//                 })
//             }
//         }
//         // Iterator is not available or exhausted. We need to resume a search for a match.
//         let mut node;
//         let mut cursor;
//         let k_n_1 = K::max_value().as_() - 1; // max of k - 1
//         let mut result = None;
//         while self.nodes.len() > 0 {
//             node = self.nodes.last().unwrap();
//             cursor = self.cursors.last_mut().unwrap();

//             // Need to iterate through each child unless it's container type
//             match node.child(&K::from_usize(*cursor).unwrap()) {
//                 NodeType::Trie(t) => {
//                     self.prefixes.push(K::from_usize(*cursor).unwrap());
//                     self.level += 1;
//                     if self.level >= self.query.len() {
//                         // The matched trie has entire query as prefix and
//                         // the trie is descendant of that trie node

//                         // Since query is exhausted, everything in this trie node
//                         // must be examine
//                         self.cursors.push(0);

//                         if t.value().is_some() {
//                             result = Some(StartsWith {
//                                 prefix: &[],
//                                 key: & self.query[..self.level],
//                                 value: t.value().unwrap()
//                             });
//                             // Push a node to be process into stack eagerly as we will early exit loop
//                             self.nodes.push(t);
//                             break;
//                         }
//                     } else {
//                         // Query is not yet exhausted
//                         // Remove processed trie node out
//                         self.nodes.pop(); 
//                         self.cursors.pop();

//                         // Push a next query into stack
//                         self.cursors.push(self.query[self.level].as_());
//                     }

//                     // Push a node to be process into stack
//                     self.nodes.push(t);
//                 },
//                 NodeType::Hybrid((bucket, _)) | NodeType::Pure(bucket) => {
//                     self.bucket = Some(bucket.iter());

//                     // Update self property first before attemp to iterate the bucket.
//                     // This is to ensure that if returned iterator is emptied, it won't cause
//                     // infinite recursion.
//                     if self.level > self.query.len() && *cursor < k_n_1 {
//                         // Bucket is descendant of trie node that is also descendant of prefix
//                         *cursor += 1;
//                     } else {
//                         self.level -= 1;
//                         self.prefixes.pop();
//                         self.nodes.pop();
//                         self.cursors.pop();
//                     }

//                     result = self.next(); // Recursive iterating on bucket
//                 },
//                 NodeType::None => {
//                     if self.level >= self.query.len() {
//                         if *cursor < k_n_1 { // cursor is < n - 1 element
//                             *cursor += 1;
//                         } else {
//                             // Exhausted current trie with None as last elm
//                             self.level -= 1;
//                             self.prefixes.pop();
//                             self.nodes.pop();
//                             self.cursors.pop();
//                         }
//                     } else {
//                         // Query is not prefix of key in trie
//                         return None
//                     }
//                 }
//             }
//         }
//         result
//     }
// }

#[cfg(test)]
mod tests;