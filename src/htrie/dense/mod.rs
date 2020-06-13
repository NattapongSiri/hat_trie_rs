use super::*;

use num_traits::{bounds::Bounded, AsPrimitive, FromPrimitive, Unsigned};

/// Find a element `K` that will split the given `bucket` in half at specified element index of key.
/// For example: if key is ["a", "b", "c"] and ["a", "c", "d"], given index is 1, the returned `K` will be
/// "c". If given idex is 2, the return `K` will be "d". This is because when index is 1, it look at second
/// element of both key thus it found "b" and "c". When index is 2, it look at third element which is
/// "c" and "d".
fn find_half_point<K, V>(bucket: &ArrayHash<twox_hash::XxHash64, Box<[K]>, V>, index: usize, start: usize, end: usize) -> K 
where K: AsPrimitive<usize> + core::hash::Hash + FromPrimitive + Bounded + PartialEq + PartialOrd + Unsigned,
        Box<[K]>: Clone + core::cmp::PartialEq, 
        V: Clone {
    // Find split point
    let mut count = vec![0; end - start + 1]; // Save some memory by allocate only necessary size
    let mut split_portion = 0;

    for (k, _) in bucket.iter() {
        if k.len() > index {
            count[k[index].as_() - start] += 1; // Accumulate each individual key
            split_portion += 1; // Accumulate total number of entry
        }
    }

    split_portion /= 2; // Now we got split portion

    let mut split_point = 0;
    let mut first_portion = 0;

    while first_portion < split_portion {
        first_portion += count[split_point];
        split_point += 1;
    }

    split_point += start;

    K::from_usize(split_point).unwrap()
}

/// Construct a hat-trie hybrid container by using raw pointer to skip
/// lookup time.
/// This is possible for "dense" type only as each key element can be converted into
/// usize to directly index into child node.
/// 
/// By convention, this is safe. It is because it use raw pointer only by internal
/// operation which when mutation occur, it occur under `&mut self` of outer type thus
/// make borrow checker enforce borrow rule as a whole on outer object.
#[derive(Clone, Debug)]
struct ChildsContainer<K, T, V> 
where K: AsPrimitive<usize> + core::hash::Hash + PartialEq + PartialOrd,
      Box<[K]>: Clone + core::cmp::PartialEq, 
      T: Clone + TrieNode<K, V>,
      V: Clone {
    /// Owned childs of a node. It need to be pre-allocated to prevent re-allocation which
    /// if happen, will invalidate all pointers that point to it.
    childs: Vec<NodeType<K, T, V>>,
    _internal_pointer: Box<[*mut NodeType<K, T, V>]>
}

impl<K, T, V> ChildsContainer<K, T, V> 
where K: AsPrimitive<usize> + Bounded + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Unsigned,
      Box<[K]>: Clone + core::cmp::PartialEq, 
      T: Clone + TrieNode<K, V>,
      V: Clone + Default {
    /// Construct an empty childs container which initialize to Single "hybrid" node and
    /// all pointer in parent point to this single child.
    pub fn new(size: usize) -> Self {
        let mut childs = Vec::with_capacity(size);
        let start = K::min_value().as_();
        let end = K::max_value().as_();
        let bucket = NodeType::Hybrid((ArrayHashBuilder::default().build(), K::from_usize(start).unwrap()..=K::from_usize(end).unwrap()));
        childs.push(bucket);
        let ptrs = (0..size).map(|_| (&mut childs[0]) as *mut NodeType<K, T, V>).collect();
        ChildsContainer {
            childs,
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
    fn maybe_split(&mut self, key: K) -> bool {
        // This is a new trie which will be parent of new hybrid node yield from pure node split.
        let mut pure_trie = None;
        // This is a result of hybrid node split.
        let mut split_result = None;
        match &mut self[key] {
            NodeType::Pure(bucket) => {
                if bucket.len() >= super::BURST_THRESHOLD {
                    // flag that current key need to be burst/split
                    pure_trie = Some(key);
                } else {
                    return false
                }
            },
            NodeType::Hybrid((bucket, range)) => {
                if bucket.len() >= super::BURST_THRESHOLD {
                    // Need to split
                    let start = range.start().as_();
                    let end = range.end().as_();

                    // Find split point
                    let split_point = find_half_point(bucket, 0, start, end + 1);
                    // Now we got split point

                    // Make a new bucket to store half of old bucket
                    let new_bucket = bucket.split_by(|(k, _)| {
                        k[0] >= split_point
                    });

                    *range = K::from_usize(start).unwrap()..=K::from_usize(split_point.as_() - 1).unwrap();

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
                    let trie = NodeType::Trie(T::new_split(bucket));
                    // Place new child at K
                    std::mem::replace(&mut *self._internal_pointer[k], trie);
                }
            }
            return true
        } else if let Some((new_bucket, [start, split_point, end])) = split_result {
            // Post processing for Hybrid split.
            // 1. change existing bucket to pure if needed. 2. update parent pointers for new bucket
            if start == split_point {
                // The only case where we need to make node Pure
                let old = std::mem::take(&mut self[key]);
                // The only possible type in here is Hybrid
                if let NodeType::Hybrid((table, _)) = old {
                    // Range can only be one here so we ignore it.

                    // Now replace the old Hybrid with Pure type
                    std::mem::replace(&mut self[key], NodeType::Pure(table));
                }
            }

            if split_point < end {
                self.childs.push(NodeType::Hybrid((new_bucket, K::from_usize(split_point).unwrap()..=K::from_usize(end).unwrap())));
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

impl<K, T, V> core::ops::Index<K> for ChildsContainer<K, T, V> 
where K: AsPrimitive<usize> + core::hash::Hash + PartialEq + PartialOrd + Unsigned,
      Box<[K]>: Clone + core::cmp::PartialEq, 
      T: Clone + TrieNode<K, V>,
      V: Clone {
    type Output=NodeType<K, T, V>;

    fn index(&self, idx: K) -> &Self::Output {
        unsafe {
            &*self._internal_pointer[idx.as_()]
        }
    }
}

impl<K, T, V> core::ops::IndexMut<K> for ChildsContainer<K, T, V> 
where K: AsPrimitive<usize> + core::hash::Hash + PartialEq + PartialOrd + Unsigned,
      Box<[K]>: Clone + core::cmp::PartialEq, 
      T: Clone + TrieNode<K, V>,
      V: Clone {

    fn index_mut(&mut self, idx: K) -> &mut Self::Output {
        unsafe {
            &mut *self._internal_pointer[idx.as_()]
        }
    }
}

#[derive(Clone, Debug)]
pub struct DenseVecTrieNode<K, V> 
where K: AsPrimitive<usize> + Bounded + Copy + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
      Box<[K]>: Clone + PartialEq,
      V: Clone + Default {
    childs: ChildsContainer<K, Self, V>,
    value: Option<V>,
}

impl<K, V> TrieNode<K, V> for DenseVecTrieNode<K, V> 
where K: AsPrimitive<usize> + Bounded + Copy + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
      Box<[K]>: Clone + PartialEq,
      V: Clone + Default {
    fn new_split(mut bucket: ArrayHash<twox_hash::XxHash64, Box<[K]>, V>) -> Self {
        let start = K::min_value().as_();
        let end = K::max_value().as_();
        let mut old_size = bucket.len();
        let split_point = find_half_point(&bucket, 1, start, end);
        let mut node_value = None;
        let mut left = ArrayHashBuilder::default().build();
        let mut right = ArrayHashBuilder::default().build();

        for (key, value) in bucket.drain() {
            if key.len() == 1 {
                // There's only single character in key.
                // It value must be present on new Trie node and remove itself from bucket.
                node_value = Some(value);
                old_size -= 1;
                continue;
            }
            if key[1] >= split_point {
                assert!(right.put(key[1..].into(), value).is_none());
            } else {
                assert!(left.put(key[1..].into(), value).is_none());
            }
        }
        assert_eq!(old_size, left.len() + right.len());

        // Construct a child container manually as we need to properly align each
        // K with correct side of bucket.
        // In future, if other struct also need this, we shall move this to ChildsContainer struct
        let mut childs = vec![
            NodeType::Hybrid((left, K::from_usize(start).unwrap()..=K::from_usize(split_point.as_() - 1).unwrap())), 
            NodeType::Hybrid((right, split_point..=K::from_usize(end).unwrap())) ];

        let split_point_usize = split_point.as_();
        let ptr = (start..=end).map(|key| {
            if key >= split_point_usize {
                (&mut childs[1]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
            } else {
                (&mut childs[0]) as *mut NodeType<K, DenseVecTrieNode<K, V>, V>
            }
        }).collect::<Vec<*mut NodeType<K, DenseVecTrieNode<K, V>, V>>>().into_boxed_slice();
        
        DenseVecTrieNode {
            childs: ChildsContainer {
                childs,
                _internal_pointer: ptr
            },
            value: node_value,
        }
    }

    fn child<'a>(&'a self, key: &K) -> &'a NodeType<K, Self, V> {
        &self.childs[*key]
    }

    fn value(&self) -> Option<&V> {
        self.value.as_ref()
    }

    fn value_mut(&mut self) -> &mut Option<V> {
        &mut self.value
    }

    fn put(&mut self, key: &[K], value: V) -> Option<V> {
        if key.len() == 0 {return None} // Cannot put empty key into trie
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
                    bucket.put(key[offset..].into(), value);
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
                    nodes.maybe_split(key[offset]);
                },
                NodeType::Pure(childs) => {
                    // For Pure type, it's `ArrayHash` that contains whole remaining key.
                    // We can simply leverage `put` method using remaining key and value.
                    let old = childs.put(key[offset..].into(), value);
                    
                    if let Some((_, v)) = old {
                        return Some(v)
                    } else {
                        return None
                    }
                },
                NodeType::Hybrid(bucket) => {
                    // For Hybrid type, it has left and right and split point to determine
                    // which `ArrayHash` to put the remaining key and value.
                    let old = bucket.0.put(key[offset..].into(), value);
                    if let Some((_, v)) = old {
                        return Some(v)
                    } else {
                        return None
                    }
                }
            }
        }
    }

    fn try_put<'a>(&'a mut self, key: &[K], value: V) -> Option<&'a V> where K: 'a {
        if key.len() == 0 {return None} // Cannot put empty key into trie
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
                        bucket.put(key[offset..].into(), value);
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
                        return childs.try_put(key[offset..].into(), value)
                    },
                    NodeType::Hybrid(bucket) => {
                        // For Hybrid type, it has left and right and split point to determine
                        // which `ArrayHash` to try put the remaining key and value.
                        return bucket.0.try_put(key[offset..].into(), value)
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
}

impl<K, V> DenseVecTrieNode<K, V> 
where K: Copy + AsPrimitive<usize> + Bounded + core::fmt::Debug + core::hash::Hash + FromPrimitive + PartialEq + PartialOrd + Sized + Unsigned,
      Box<[K]>: Clone + PartialEq,
      V: Clone + Default {

    pub fn new() -> DenseVecTrieNode<K, V> {
        DenseVecTrieNode {
                // root always has no value
            value: None,
            // Allocate Trie with length equals to Trie size. Since K is not `Clone`, we need to manually create all of it.
            childs: ChildsContainer::new(2usize.pow(std::mem::size_of::<K>() as u32 * 8u32))
        }
    }
}

#[cfg(test)]
mod tests;