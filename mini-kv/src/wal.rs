use super::util::{read_pairs_from_buffer, write_pair_to_file};
use anyhow::Result;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;

pub struct WAL {
    util_file: Box<Path>,
    file: Option<File>,
}

impl WAL {
    pub fn new(util_file: Box<Path>) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&util_file)
            .ok();
        WAL { util_file, file }
    }

    pub fn write(&mut self, key: &str, value: &str) -> Result<()> {
        if let Some(file) = self.file.as_mut() {
            write_pair_to_file(key, value, file)?;
        }
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        self.file = None;
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.util_file)?;
        self.file = Some(file);
        Ok(())
    }

    pub fn restore(&mut self) -> Result<HashMap<String, String>> {
        self.file = None;
        let mut file = OpenOptions::new().read(true).open(&self.util_file)?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let pairs = read_pairs_from_buffer(&buffer);

        let mut hm = HashMap::new();
        pairs.unwrap().into_iter().for_each(|(k, v)| {
            hm.insert(k, v);
        });

        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.util_file)
            .ok();

        Ok(hm)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_wal() {
        let mut db = Cyrus::test("wal_db");

        // Insert values
        db.insert("key1", "value1").unwrap();
        db.insert("somerandomkey", "longertestvalue").unwrap();
        let wal = db.get_wal();
        let restored_data = wal.restore().unwrap();

        // Ensure inserted values are correct
        assert_eq!(restored_data.get("key1").unwrap(), "value1");
        assert_eq!(
            restored_data.get("somerandomkey").unwrap(),
            "longertestvalue"
        );

        // Ensure tombstone overwrites prev
        db.remove("key1").unwrap();
        let wal = db.get_wal();
        let restored_data = wal.restore().unwrap();
        assert_eq!(restored_data.get("key1").unwrap(), CYRUS_TOMB);

        // Ensure clear empties WAL
        wal.clear().unwrap();
        assert_eq!(wal.restore().unwrap().len(), 0);
    }
}
