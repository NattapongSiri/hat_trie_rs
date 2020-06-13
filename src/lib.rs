mod htrie;

pub use crate::htrie::{TrieNode, dense::DenseVecTrieNode};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
