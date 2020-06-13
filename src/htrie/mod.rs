use ahtable::*;

pub mod dense;

const BURST_THRESHOLD: usize = 16384;

#[derive(Clone, Debug)]
pub enum NodeType<K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: TrieNode<K, V>,
      V: Clone {
    None,
    Trie(T),
    Pure(ArrayHash<twox_hash::XxHash64, Box<[K]>, V>),
    Hybrid((ArrayHash<twox_hash::XxHash64, Box<[K]>, V>, core::ops::RangeInclusive<K>))
}

impl<K, T, V> Default for NodeType<K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: TrieNode<K, V>,
      V: Clone {
        fn default() -> Self {
            NodeType::None
        }
}

pub trait TrieNode<K, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      V: Clone,
      Self: Sized {
    /// Construct a new trie node from a bucket of pure node.
    /// This happen whenever burst or split threshold reach.
    /// It consume a given bucket and return a new Trie node which contains
    /// splitted bucket as childs.
    fn new_split(bucket: ArrayHash<twox_hash::XxHash64, Box<[K]>, V>) -> Self;
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
    
    /// Get a value from this trie associated with given key slice.
    fn get<'a>(&'a self, key: &[K]) -> Option<&'a V> where K: 'a {
        if key.len() == 0 {return None} // Empty key won't exist in such trie
        let mut offset = 0; // Progress of key being process
        // let mut nodes = &self.child(key); // Set root childs as nodes to be search for a key
        let mut parent = self;
        let mut node;

        loop {
            // Dense Trie can use byte as index into each possible slot
            node = parent.child(&key[offset]);

            if offset < key.len() {
                // if there's some remain key to be process, we analyze child of current node
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
            cur_len: 1,
            node: self,
            query: key
        }
    }

    /// Find a longest prefix from given key from this Trie.
    /// 
    /// This is a utility function that perform exactly like you iterate on method 
    /// [prefix](struct.TrieNode.html#method.prefix) to find a longest prefix by yourself.
    fn longest_prefix<'a, 'b>(&'a self, key: &'b [K]) -> Option<(&'b[K], &'a V)> where K: 'a {
        let mut longest: Option<(&[K], &V)> = None;
        self.prefix(key).for_each(|(key, value)| {
            if let Some((k, _)) = longest {
                if key.len() > k.len() {
                    longest = Some((key, value));
                }
            } else {
                longest = Some((key, value));
            }
        });

        longest
    }
}

#[derive(Debug)]
pub struct PrefixIterator<'a, 'b, K, T, V> 
where K: Copy + core::hash::Hash + PartialEq + PartialOrd + Sized,
      Box<[K]>: Clone + PartialEq,
      T: 'a + TrieNode<K, V>,
      V: Clone {
    _phantom: core::marker::PhantomData<V>,
    bucket: Option<&'a ahtable::ArrayHash<twox_hash::XxHash64, Box<[K]>, V>>,
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
        if let Some(ref bucket) = self.bucket {
            for i in (self.cursor + self.cur_len)..=self.query.len() {
                let cur_key = &self.query[self.cursor..i];
                if let Some(v) = bucket.smart_get(cur_key) {
                    self.cur_len = i - self.cursor + 1;
                    return Some((&self.query[..i], v))
                }
            }

            // It is impossible to find value in any other bucket because of the type of data structure
            return None
        } else if self.query.len() == 0 {
            return None
        } else {
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
                    },
                    NodeType::Hybrid((ref bucket, _)) => {
                        self.bucket = Some(bucket);
                    }
                }
            }

            if let Some(ref bucket) = self.bucket {
                for i in (self.cursor + self.cur_len)..self.query.len() {
                    let cur_key = &self.query[self.cursor..i];
                    if let Some(v) = bucket.smart_get(cur_key) {
                        self.cur_len = i - self.cursor + 1;
                        return Some((&self.query[..i], v))
                    }
                }
                return None
            }
        }
        None
    }
}