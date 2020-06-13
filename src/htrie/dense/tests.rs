use super::*;

#[test]
fn string_dense_trie() {
    let mut dvtn = DenseVecTrieNode::new();
    assert!(dvtn.get("".as_bytes()).is_none());
    assert_eq!(dvtn.put("Yo".as_bytes(), 1), None);
    assert_eq!(dvtn.get("Yo".as_bytes()), Some(&1));
    assert_eq!(dvtn.put("Yes".as_bytes(), 2), None);
    // dbg!(&dvtn);
    assert_eq!(dvtn.get("Yes".as_bytes()), Some(&2));
}

#[test]
fn i16_dense_trie() {
    let mut dvtn: DenseVecTrieNode<u16, usize> = DenseVecTrieNode::new();
    assert!(dvtn.get(&[1u16, 2]).is_none());
    assert_eq!(dvtn.put(&[1, 2], 1), None);
    assert_eq!(dvtn.get(&[1, 2]), Some(&1));
    assert_eq!(dvtn.put(&[1, 2, 3], 2), None);
    assert_eq!(dvtn.get(&[1, 2, 3]), Some(&2));
    assert_eq!(dvtn.put(&[2, 2, 3], 3), None);
    assert_eq!(dvtn.get(&[2, 2, 3]), Some(&3));
}

#[test]
fn test_find_half() {
    let mut ah = ArrayHashBuilder::default().build();
    let mut keys : Vec<Box<[u8]>> = Vec::with_capacity(1000);
    for i in 0u8..10 {
        for j in 0u8..12 {
            for k in 0u8..14 {
                keys.push(Box::new([i, j, k]));
            }
        }
    }
    for (i, k) in keys.iter().enumerate() {
        ah.put(k.clone(), i);
    }
    assert_eq!(5, find_half_point(&ah, 0, 0, 10));
    assert_eq!(6, find_half_point(&ah, 1, 0, 12));
    assert_eq!(7, find_half_point(&ah, 2, 0, 14));
}

#[test]
fn test_hybrid_2_2hybrid_split() {
    let mut trie = DenseVecTrieNode::new();
    let mut keys: Vec<Box<[u8]>> = Vec::with_capacity(crate::htrie::BURST_THRESHOLD + 1);
    // First split is always hybrid split because first node is always hybrid.
    for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
        let mut key = vec![1u8];
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }
    for i in (crate::htrie::BURST_THRESHOLD / 2 + 1)..=(crate::htrie::BURST_THRESHOLD + 1) {
        let mut key = vec![2u8];
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }

    // Attempt to put all entry into trie which will goes 1 element beyond threshold so 
    // when last element is added, it shall trigger bucket split.
    for i in 0..keys.len() {
        trie.put(&*keys[i], i);
    }

    // Retrieve back all the key to verify integrity of trie
    for i in 0..keys.len() {
        if let Some(value) = trie.get(&*keys[i]) {
            assert_eq!(*value, i);
        } else {
            panic!("Key {:?} is missing from trie", &*keys[i]);
        }
    }
}
#[test]
fn test_hybrid_2_purehybrid_split() {
    let mut trie = DenseVecTrieNode::new();
    let mut keys: Vec<Box<[u8]>> = Vec::with_capacity(crate::htrie::BURST_THRESHOLD + 1);
    // First split is always hybrid split because first node is always hybrid.
    for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
        let mut key = vec![0u8];
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }
    for i in (crate::htrie::BURST_THRESHOLD / 2 + 1)..=(crate::htrie::BURST_THRESHOLD + 1) {
        let mut key = vec![1u8];
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }

    // Attempt to put all entry into trie which will goes 1 element beyond threshold so 
    // when last element is added, it shall trigger bucket split.
    for i in 0..keys.len() {
        trie.put(&*keys[i], i);
    }

    // Retrieve back all the key to verify integrity of trie
    for i in 0..keys.len() {
        if let Some(value) = trie.get(&*keys[i]) {
            assert_eq!(*value, i);
        } else {
            panic!("Key {:?} is missing from trie", &*keys[i]);
        }
    }
}
#[test]
fn test_hybrid_2_hybridpure_split() {
    let mut trie = DenseVecTrieNode::new();
    let mut keys: Vec<Box<[u8]>> = Vec::with_capacity(crate::htrie::BURST_THRESHOLD + 1);
    // First split is always hybrid split because first node is always hybrid.
    for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
        let mut key = vec![254u8];
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }
    for i in (crate::htrie::BURST_THRESHOLD / 2 + 1)..=(crate::htrie::BURST_THRESHOLD + 1) {
        let mut key = vec![255u8];
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }

    // Attempt to put all entry into trie which will goes 1 element beyond threshold so 
    // when last element is added, it shall trigger bucket split.
    for i in 0..keys.len() {
        trie.put(&*keys[i], i);
    }

    // Retrieve back all the key to verify integrity of trie
    for i in 0..keys.len() {
        if let Some(value) = trie.get(&*keys[i]) {
            assert_eq!(*value, i);
        } else {
            panic!("Key {:?} is missing from trie", &*keys[i]);
        }
    }
}

#[test]
fn test_hybrid_2_purepure_split() {
    let mut trie = DenseVecTrieNode::new();

    let mut keys: Vec<Box<[u8]>> = Vec::with_capacity(crate::htrie::BURST_THRESHOLD * 256 + 1);
    // The simplest way to have hybrid to split into two pure half is to repeatly add the node
    // where prefix range from 0-255 and each of it has size equals to half of threshold...
    // It gonna take a while as it require about 4 million entry
    for pref in 0u8..=255 {
        // Create all possible prefix[0] of key so that in the end, we will get all pure node
        for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
            let mut key = vec![pref];
            let mut j = i;
            while j > 0 { // Add element until it is unique
                key.push((j % 8) as u8);
                j /= 8;
            }
            keys.push(key.into_boxed_slice());
        }
    }

    // Attempt to put all entry into trie. The last element shall trigger a split from hybrid into two pure.
    for i in 0..keys.len() {
        trie.put(&*keys[i], i);
    }

    // Retrieve back all the key to verify integrity of trie
    for i in 0..keys.len() {
        if let Some(value) = trie.get(&*keys[i]) {
            assert_eq!(*value, i);
        } else {
            panic!("Key {:?} is missing from trie", &*keys[i]);
        }
    }
}

#[test]
fn test_pure_2_hybrid_split() {
    let mut trie = DenseVecTrieNode::new();

    let mut keys: Vec<Box<[u8]>> = Vec::with_capacity(crate::htrie::BURST_THRESHOLD * 256 + 1);
    for pref in 254..=255 {
        // We create prefix at extreme highest end
        for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
            let mut key = vec![pref];
            let mut j = i;
            while j > 0 { // Add element until it is unique
                key.push((j % 8) as u8);
                j /= 8;
            }
            keys.push(key.into_boxed_slice());
        }
    }

    let longest = keys.len() - 1; // This key shall be longest one so append anything to it shall make a unique key

    // We duplicate prefix at extreme highest end to trigger pure split
    for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
        let mut key = keys[longest].clone().into_vec();
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }

    // Attempt to put all entry into trie. The last element shall trigger a split from hybrid into two pure.
    for i in 0..keys.len() {
        trie.put(&*keys[i], i);
    }

    // Retrieve back all the key to verify integrity of trie
    for i in 0..keys.len() {
        if let Some(value) = trie.get(&*keys[i]) {
            assert_eq!(*value, i);
        } else {
            panic!("Key {:?} is missing from trie", &*keys[i]);
        }
    }
}

#[test]
fn test_prefix() {
    fn lifetime_check<'a>(key: &'a [u8]) -> Vec<&'a [u8]> {
        let mut dvtn: DenseVecTrieNode<u8, u8> = DenseVecTrieNode::new();
        dvtn.put(&[1u8], 1u8);
        dvtn.put(&[1u8, 2, 3], 2u8);
        dvtn.put(&[1u8, 2, 3, 4], 3u8);
        dvtn.prefix(key).map(|(key, _)| {key}).collect()
    }

    assert_eq!(lifetime_check(&[0u8, 1]).len(), 0);
    let key = &[1u8, 1];
    let pf = lifetime_check(key);
    assert_eq!(pf.len(), 1);
    assert_eq!(pf[0], &key[..1]);

    let key = &[1u8, 2, 3, 5, 6];
    let pf = lifetime_check(key);
    
    assert_eq!(pf.len(), 2);
    for k in pf {
        assert_eq!(&key[..k.len()], k);
    }
}

#[test]
fn test_layered_trie_prefix() {
    let mut keys: Vec<Box<[u8]>> = Vec::with_capacity(crate::htrie::BURST_THRESHOLD * 256 + 1);
    // Test extract a prefix key path on trie branch that has trie sub layer
    for pref in 244u8..=255 {
        // Create all possible prefix[0] of key so that in the end, we will get all pure node
        for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
            let mut key = vec![pref];
            let mut j = i;
            while j > 0 { // Add element until it is unique
                key.push((j % 8) as u8);
                j /= 8;
            }
            keys.push(key.into_boxed_slice());
        }
    }

    let longest = keys.len() - 1; // This key shall be longest one so append anything to it shall make a unique key

    // We duplicate prefix at extreme highest end to trigger pure split.
    // Now we have a trie node that point to trie then hybrid node
    for i in 1..=(crate::htrie::BURST_THRESHOLD / 2 + 1) {
        let mut key = keys[longest].clone().into_vec();
        let mut j = i;
        while j > 0 { // Add element until it is unique
            key.push((j % 8) as u8);
            j /= 8;
        }
        keys.push(key.into_boxed_slice());
    }

    let mut dvt = DenseVecTrieNode::new();

    keys.into_iter().enumerate().for_each(|(i, k)| {dvt.put(&*k, i);});
    // iterate to gather all prefix of some specific value
    let test_key = &[255u8, 
                        1,
                        0,
                        0,
                        0,
                        2,
                        2,
                        0,
                        0,
                        2,
                        3];
    let mut expected_key: Vec<&[u8]> = vec![];
    if let NodeType::Trie(ref t) = dvt.child(&255u8) {
        if t.value().is_some() {
            expected_key.push(&test_key[..1]);
        }
        if let NodeType::Hybrid((bucket, _)) = t.child(&1u8) {
            for i in 2..test_key.len() {
                if let Some(_) = bucket.smart_get(&test_key[1..i]) {
                    expected_key.push(&test_key[..i]);
                }
            }
        } else {
            panic!("Missing expected bucket for 255u8->2u8")
        }
    } else {
        panic!("Missing expected trie node, 255u8")
    }

    // verify that prefix method return the all value found from iterate
    dvt.prefix(test_key).for_each(|(key, _)| {
        if let Some(index) = expected_key.iter().position(|e| {*e == key}) {
            expected_key.remove(index);
        } else {
            panic!("{:?} is return but it is not expected prefix", key);
        }
    });

    assert_eq!(expected_key, Vec::<&[u8]>::new(), "Not all expected keys are found");
}
#[test]
fn test_longest_prefix() {
    let key = &[1u8, 2, 3, 5, 6];
    fn test_lifetime(key: &[u8]) -> &[u8] {
        let mut dvtn: DenseVecTrieNode<u8, u8> = DenseVecTrieNode::new();
        dvtn.put(&[1u8], 1u8);
        dvtn.put(&[1u8, 2, 3], 2u8);
        dvtn.put(&[1u8, 2, 3, 4], 3u8);
        dvtn.longest_prefix(key).unwrap().0
    }

    let key = test_lifetime(key);
    
    // if let Some((key, value)) = pf {
        assert_eq!(key, &[1u8, 2, 3]);
    //     assert_eq!(*value, 2u8);
    // } else {
    //     panic!("Cannot find longest prefix")
    // }
}