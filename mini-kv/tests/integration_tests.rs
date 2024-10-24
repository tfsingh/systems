use rand::Rng;
use std::time::Instant;
use std::{collections::HashMap, fs};

use cyrus::Cyrus;

#[test]
fn test_simple_ops() {
    let _ = fs::remove_dir_all("dbs/test_db");
    let mut db = Cyrus::new("simple_db");
    db.insert("test", "value").unwrap();
    assert_eq!("value", &db.get("test").unwrap());
    db.remove("test").unwrap();
    assert_eq!(None, db.get("test"));
}

#[test]
fn test_stress_ops_random_values_overwrite() {
    let _ = fs::remove_dir_all("dbs/stress_overwrite_db");
    let num_operations: usize = 100_000;
    let mut db = Cyrus::new("stress_overwrite_db");
    let mut hash_map = HashMap::new();
    let mut rng = rand::thread_rng();
    let key_range = 1000;

    for _ in 0..num_operations {
        let key = format!("key_{}", rng.gen_range(0..key_range));

        let value_len = rng.gen_range(1..=100);
        let value: String = (0..value_len)
            .map(|_| rng.gen_range(b'a'..=b'z') as char)
            .collect();

        db.insert(&key, &value).unwrap();
        hash_map.insert(key.clone(), value.clone());
    }

    for i in 0..key_range {
        let key = format!("key_{}", i);
        let value = hash_map.get(&key);
        assert_eq!(value, db.get(&key).as_ref());
    }
}

#[test]
fn test_stress_ops_correctness() {
    let _ = fs::remove_dir_all("dbs/stress_correctness_db");
    let num_operations: usize = 100_000;

    let mut db = Cyrus::new("stress_correctness_db");
    let mut hash_map = HashMap::new();

    for i in 0..num_operations {
        let key = format!("key_{}", i);
        let value = format!("value_{}", i);

        db.insert(&key, &value).unwrap();
        hash_map.insert(key.clone(), value.clone());

        assert_eq!(Some(&value), db.get(&key).as_ref());
        assert_eq!(Some(&value), hash_map.get(&key));

        db.remove(&key).unwrap();
        hash_map.remove(&key);

        assert_eq!(None, db.get(&key));
        assert_eq!(None, hash_map.get(&key));
    }
}

#[test]
fn test_performance() {
    let _ = fs::remove_dir_all("dbs/performance_db");
    let num_writes: usize = 1_000_000;
    let num_reads: usize = 2500;
    let mut db = Cyrus::new("performance_db");

    let start_writes = Instant::now();

    for i in 0..num_writes {
        let key = format!("key_{}", i);
        let value = format!("value_{}", i);

        db.insert(&key, &value).unwrap();
    }

    let duration_writes = start_writes.elapsed();
    println!(
        "Time taken for {} writes: {:?}",
        num_writes, duration_writes
    );

    let start_reads = Instant::now();

    for i in (0..num_reads).step_by(num_writes / num_reads) {
        let key = format!("key_{}", i);
        let _ = db.get(&key);
    }

    let duration_reads = start_reads.elapsed();
    println!("Time taken for {} reads: {:?}", num_reads, duration_reads);
}
