use core::hash::Hash;
use crate::htrie::TrieKey;

impl<T> TrieKey for [T] where T: Clone + Hash + PartialEq + PartialOrd {
    fn len(&self) -> usize {
        self.len()
    }
}