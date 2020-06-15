use ahtable::*;

pub mod dense;

const BURST_THRESHOLD: usize = 16384;

/// All possible type of hat-trie node.
#[derive(Clone, Debug)]
pub enum NodeType<K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: TrieNode<K, V>,
      V: Clone {
    /// Empty type. This is only used for temporary value replacement.
    /// For example, when splitting node.
    None,
    /// Sub layer of trie which is also an object of type trie.
    Trie(T),
    /// Pure container type. It can have only one parent trie node.
    Pure(ArrayHash<twox_hash::XxHash64, Box<[K]>, V>),
    /// Hybrid contaainer type. It can have more than one parent trie node.
    Hybrid((ArrayHash<twox_hash::XxHash64, Box<[K]>, V>, core::ops::RangeInclusive<K>))
}

/// Implement Default so that we can swap value out to perform node bursting/splitting.
impl<K, T, V> Default for NodeType<K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: TrieNode<K, V>,
      V: Clone {
        fn default() -> Self {
            NodeType::None
        }
}

/// A TrieNode operation where every implementation of Trie in this crate shall support.
pub trait TrieNode<K, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      V: Clone,
      Self: Sized {
    /// Construct a new trie node from a bucket of pure node.
    /// This happen whenever burst or split threshold reach.
    /// It consume a given bucket and return a new Trie node which contains
    /// splitted bucket as childs.
    /// 
    /// The new trie node will have specified burst/split threshold.
    fn new_split(bucket: ArrayHash<twox_hash::XxHash64, Box<[K]>, V>, threshold: usize) -> Self;
    /// Get a child node of given key from current node.
    fn child(&'_ self, key: &K) -> &'_ NodeType<K, Self, V>;
    /// Retrieve a value of current node.
    fn value(&self) -> Option<&V>;
    /// Retrieve a mutable value of current node.
    fn value_mut(&mut self) -> &mut Option<V>;
    /// Put value into this trie node.
    /// 
    /// If key is already exist in this trie, it replace existing entry with give
    /// key/value and return existing value to caller.
    fn put(&mut self, key: &[K], value: V) -> Option<V>;
    /// Try putting value into this trie.
    /// 
    /// If key is already exist in this trie, it will return existing value to caller without
    /// any change to the trie.
    fn try_put<'a>(&'a mut self, key: &[K], value: V) -> Option<&'a V> where K: 'a;

    /// Utilities function that help shortcut key lookup when
    /// caller request key greater than max_key_len, it mean there's no such key
    /// in this trie.
    /// 
    /// By default, it return MAX value of usize.
    #[inline(always)]
    fn max_key_len(&self) -> usize {
        usize::MAX
    }
    /// Utilities function that help shortcut key lookup when
    /// caller request key less than min_key_len, it mean there's no such key
    /// in this trie.
    /// 
    /// By default, it return MIN value of usize.
    #[inline(always)]
    fn min_key_len(&self) -> usize {
        usize::MIN
    }
    
    /// Get a value from this trie associated with given key slice.
    fn get<'a>(&'a self, key: &[K]) -> Option<&'a V> where K: 'a {
        
        // Key that is not part of this trie
        match key.len() {
            0 => return None,
            x if x < self.min_key_len() => return None,
            x if x > self.max_key_len() => return None,
            _ => ()
        } 

        let mut offset = 0; // Progress of key being process
        // let mut nodes = &self.child(key); // Set root childs as nodes to be search for a key
        let mut parent = self;
        let mut node;

        loop {

            if offset < key.len() {
                // if there's some remain key to be process, we analyze child of current node
                // Dense Trie can use byte as index into each possible slot
                node = parent.child(&key[offset]);

                match node {
                    NodeType::None => {
                        return None
                    }
                    NodeType::Trie(t) => {
                        // For simple trie, we just add all childs back to node waiting to be process
                        parent = t;
                    },
                    NodeType::Pure(child) | NodeType::Hybrid((child, _)) => {
                        // For Pure/Hybrid type, it's `ArrayHash` that contains whole remaining key.
                        // We can simply leverage `get` method using remaining key.
                        return child.get(&key[offset..].into())
                    }
                }
            } else {
                return parent.value()
            }
            offset += 1; // Move offset so next time key being use will be those that is not yet processed.
        }
    }

    /// Find all possible prefix in given key that match with this trie node.
    /// 
    /// # Parameter
    /// `key` - a slice of key to find a prefix.
    /// 
    /// # Return
    /// It return [PrefixIterator](struct.PrefixIterator.html) which can be used to 
    /// access tuple of prefix slice and value of node in this trie.
    fn prefix<'a, 'b>(&'a self, key: &'b [K]) -> PrefixIterator<'a, 'b, K, Self, V> {
        PrefixIterator {
            _phantom: core::marker::PhantomData,
            bucket: None,
            cursor: 0,
            cur_len: self.min_key_len(), // We shall reduce comparison by start at shortest key in this trie.
            node: self,
            // We only need to check prefix up to longest key in this trie.
            query: &key[..core::cmp::min(self.max_key_len(), key.len())] 
        }
    }

    /// Find a longest prefix from given key from this Trie.
    /// 
    /// It traverse the trie until either it reach container node or key is depleted.
    /// If it reaches container type first, it will start attempt to get the remain of key
    /// from the bucket and keep shrinking key until it found value.
    /// If the key is depleted first, it check if last traversed node has value.
    /// 
    /// # Parameter
    /// `key` - A key to look for longest prefix with this trie.
    /// 
    /// # Return
    /// It return `Some((key_prefix, &value))` where `key_prefix` is the longest prefix found.
    /// It return `None` if there's no prefix found.
    fn longest_prefix<'a, 'b>(&'a self, key: &'b [K]) -> Option<(&'b[K], &'a V)> where K: 'a {
        if key.len() == 0 {return None}
        let mut node = self;
        let mut i = 0;
        let mut end = core::cmp::min(key.len(), self.max_key_len());
        while i < end {
            match node.child(&key[i]) {
                NodeType::None => return None,
                NodeType::Trie(t) => {
                    node = t;
                    i += 1;
                },
                NodeType::Hybrid((bucket, _)) | NodeType::Pure(bucket) => {
                    while end > i {
                        if let Some(v) = bucket.smart_get(&key[i..end]) {
                            return Some((&key[..end], v))
                        }
                        end -= 1;
                    }

                    return None
                }
            }
        }
        if let Some(v) = node.value() {
            Some((&key[..i], v))
        } else {
            None
        }
    }
}

/// An iterator that return prefix of given query.
/// 
/// This iterator also return an exact match to query. It doesn't guarantee order of return prefix.
#[derive(Debug)]
pub struct PrefixIterator<'a, 'b, K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: 'a + TrieNode<K, V>,
      V: Clone {
    _phantom: core::marker::PhantomData<V>,
    bucket: Option<&'a ArrayHash<twox_hash::XxHash64, Box<[K]>, V>>,
    // bucket: Option<ahtable::ArrayHashIterator<'a, Box<[K]>, V>>,
    cursor: usize,
    cur_len: usize,
    node: &'a T,
    query: &'b [K]
}

impl<'a, 'b, K, T, V>  Iterator for PrefixIterator<'a, 'b, K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: 'a + TrieNode<K, V>,
      V: Clone {
    type Item=(&'b[K], &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref mut bucket) = self.bucket {
            // Resume iterating over query and use smart_get method to find if such key exist
            for i in (self.cursor + self.cur_len)..=self.query.len() {
                let cur_key = &self.query[self.cursor..i];
                if let Some(v) = bucket.smart_get(cur_key) {
                    self.cur_len = i - self.cursor + 1;
                    return Some((&self.query[..i], v))
                }
            }

            // below code use iterative style to fetch for prefix

            // while let Some((key, value)) = bucket.next() {
            //     let bound = self.cursor + key.len();
            //     if bound <= self.query.len() && &self.query[self.cursor..bound] == &**key {
            //         return Some((&self.query[..bound], value))
            //     }
            // }

            // It is impossible to find value in any other bucket because of the type of data structure
            return None
        } else if self.query.len() == 0 {
            return None
        } else {
            // Traverse the trie to find either a trie node with value or
            // a container node which will require some other way to find a prefix.
            while self.bucket.is_none() {
                match self.node.child(&self.query[self.cursor]) {
                    NodeType::None => {
                        return None
                    },
                    NodeType::Trie(ref t) => {
                        self.node = t;
                        self.cursor += 1;

                        if t.value().is_some() {
                            return Some((&self.query[..=self.cursor], t.value().as_ref().unwrap()))
                        }
                    },
                    NodeType::Pure(ref bucket) => {
                        self.bucket = Some(bucket);
                        // self.bucket = Some(bucket.iter());
                    },
                    NodeType::Hybrid((ref bucket, _)) => {
                        self.bucket = Some(bucket);
                        // self.bucket = Some(bucket.iter());
                    }
                }
            }

            // Mirror a code above to iterating on query element and use smart_get to find if key exist
            if let Some(ref mut bucket) = self.bucket {
                for i in (self.cursor + self.cur_len)..=self.query.len() {
                    let cur_key = &self.query[self.cursor..i];
                    if let Some(v) = bucket.smart_get(cur_key) {
                        self.cur_len = i - self.cursor + 1;
                        return Some((&self.query[..i], v))
                    }
                }

                // below code use iterative style to fetch for prefix

                // while let Some((key, value)) = bucket.next() {
                //     let bound = self.cursor + key.len();
                //     if bound <= self.query.len() && &self.query[self.cursor..bound] == &**key {
                //         return Some((&self.query[..bound], value))
                //     }
                // }
                return None
            }
        }
        None
    }
}

impl<'a, 'b, K, T, V> core::iter::FusedIterator for PrefixIterator<'a, 'b, K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: 'a + TrieNode<K, V>,
      V: Clone {

}