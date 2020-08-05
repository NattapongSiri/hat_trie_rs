# hat_trie
An implementation of Hat-Trie according to this [research](https://dl.acm.org/doi/pdf/10.5555/1273749.1273761) by Nikolas Askitis and Ranjan Sinha.

Some part of codes are inspired by [hat-trie](https://github.com/dcjones/hat-trie) written in C99.
A good introduction to Hat-trie can be found [here](https://github.com/Tessil/hat-trie).

## Short description
Hat-trie is a cache conscious kind of trie. This crate implements a "Hybrid" hat-trie. It consumes lesser storage than pure "Hashtable" as many common prefix will be extracted to common node of trie but this happen only if number of entry is large enough thus for small number of entry, it's virtually a "Hashtable" whereas a large number of entry, it's a mixture between "Hashtable" and trie.

# How to use
As dictionary:
```rust
use htrie::{DenseVecTrieNode, TrieNode};
let mut dvtn = DenseVecTrieNode::new();
assert!(dvtn.get("".as_bytes()).is_none());
assert_eq!(dvtn.put("Yo".as_bytes(), 1), None);
assert_eq!(dvtn.get("Yo".as_bytes()), Some(&1));
assert_eq!(dvtn.put("Yes".as_bytes(), 2), None);
assert_eq!(dvtn.get("Yes".as_bytes()), Some(&2));
```
As prefix lookup data structure:
```rust
use htrie::{DenseVecTrieNode, TrieNode};

let mut dvtn: DenseVecTrieNode<u8, u8> = DenseVecTrieNode::new();
dvtn.put(&[1u8], 1u8);
dvtn.put(&[1u8, 2, 3], 2u8);
dvtn.put(&[1u8, 2, 3, 4], 3u8);

assert_eq!(dvtn.prefix(&[0u8, 1]).map(|(key, _)| {key}).collect::<Vec<&[u8]>>().len(), 0);
let key = &[1u8, 1];
let pf = dvtn.prefix(key).map(|(key, _)| {key}).collect::<Vec<&[u8]>>();
assert_eq!(pf.len(), 1);
assert_eq!(pf[0], &key[..1]);

let key = &[1u8, 2, 3, 5, 6];
let pf = dvtn.prefix(key).map(|(key, _)| {key}).collect::<Vec<&[u8]>>();

assert_eq!(pf.len(), 2);
for k in pf {
    assert_eq!(&key[..k.len()], k);
}
```

# Caveat
Current implementation is closely similar to the one in research paper. In initialization of the `DenseVecTrieNode`, it pre-allocate memory at least equals to 2^bits of key element times size of `usize`. So in first example, it is 8 bits thus it pre-allocate memory about 2 ^ 8 * size of `usize` thus in 64 bits system, it took 1024 bytes + some overhead of `ahtable` and size of value. However, if your data is `u32`, the size of Unicode codepoint, you will requires at least 2 ^ 32 * size of `usize` thus 4,294,967,296 * size of `usize`. In 64 bits system, it is over 16GB for an empty `DenseVecTrieNode`.

That is the reasons why in example code above, we encoded data into bytes before putting it in the trie. It can also be possible to encode data into `u16` in case your data is encoded in UTF-16. However, for simplicity of example, we use `u8`. This is because the default encoding that Rust use is UTF-8.