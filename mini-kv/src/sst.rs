use crate::{compaction, util};

use anyhow::Result;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub struct SST {
    data_file: Box<Path>,
    num_sst_written: u8,
}

impl SST {
    pub fn new(data_file: Box<Path>) -> Self {
        SST {
            data_file,
            num_sst_written: 0,
        }
    }

    pub fn flush(&mut self, memtable: &BTreeMap<String, String>) -> Result<()> {
        if self.num_sst_written == 1 {
            util::write_map_to_existing_file(memtable, &self.data_file)?;
        } else {
            util::write_map_to_new_file(memtable, &self.data_file)?;
        }

        self.num_sst_written += 1;
        if self.num_sst_written == 2 {
            compaction::compact(0, &self.data_file)?;
            self.num_sst_written = 0;
        }

        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let parent_dir = Path::new(self.data_file.as_os_str()).parent().unwrap();
        let mut data_files: Vec<_> = fs::read_dir(parent_dir)
            .unwrap()
            .filter_map(|entry| {
                let entry = entry.unwrap();
                let file_name = entry.file_name();
                if file_name.to_str().unwrap().starts_with("data") {
                    Some(entry.path())
                } else {
                    None
                }
            })
            .collect();

        data_files.sort_by(|a, b| {
            let extract_number = |s: &str| {
                s.rsplit('_')
                    .next()
                    .and_then(|part| part.parse::<u32>().ok())
            };

            let a_name = a.file_name().unwrap().to_str().unwrap();
            let b_name = b.file_name().unwrap().to_str().unwrap();

            match (extract_number(a_name), extract_number(b_name)) {
                (Some(a_num), Some(b_num)) => a_num.cmp(&b_num),
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => a_name.cmp(b_name),
            }
        });

        for file_path in data_files {
            let find_value = |buffer: &Vec<u8>| -> Option<String> {
                util::read_pairs_from_buffer(buffer).ok().and_then(|pairs| {
                    pairs
                        .binary_search_by(|(k, _)| k.as_str().cmp(key))
                        .ok()
                        .map(|index| pairs[index].1.clone())
                })
            };

            for buffer in [
                util::read_recent_sst(&file_path).unwrap(),
                util::read_older_sst(&file_path).unwrap(),
            ]
            .into_iter()
            {
                let val = find_value(&buffer);
                if val.is_some() {
                    return val;
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom};

    use fs::OpenOptions;
    use util::read_pairs_from_buffer;

    use crate::*;

    #[test]
    fn test_sst_and_compaction() {
        // Initialize db
        let mut db = Cyrus::test("sst_db");
        db.insert("key1", "value1").unwrap();
        db.insert("obscurekey", "obscurevalue").unwrap();

        let memtable = db.get_memtable().clone();
        let sst = db.get_sst();
        sst.flush(&memtable).unwrap();

        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&sst.data_file)
            .unwrap();

        // Write first SST
        file.seek(SeekFrom::Start(8)).unwrap();
        let mut buffer = vec![];
        file.read_to_end(&mut buffer).unwrap();

        let res = read_pairs_from_buffer(&buffer).unwrap();
        let expected = vec![
            ("key1".to_owned(), "value1".to_owned()),
            ("obscurekey".to_owned(), "obscurevalue".to_owned()),
        ];
        assert_eq!(res, expected);

        // Clear DB
        db.clear_mem();
        db.insert("something", "else").unwrap();
        db.insert("key1", "value2").unwrap();

        let memtable = db.get_memtable().clone();
        let sst = db.get_sst();
        sst.flush(&memtable).unwrap();

        // Write second SST, trigger compaction
        let mut buffer = [0; 8];
        file.seek(SeekFrom::Start(0)).unwrap();
        file.read_exact(&mut buffer).unwrap();
        let offset = u64::from_le_bytes(buffer);
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut buffer = vec![];
        file.read_to_end(&mut buffer).unwrap();

        let res = read_pairs_from_buffer(&buffer).unwrap();
        let expected = vec![
            ("key1".to_owned(), "value2".to_owned()),
            ("something".to_owned(), "else".to_owned()),
        ];
        assert_eq!(res, expected);

        // Check get is valid
        db.clear_mem();
        let res = db.get("key1");
        assert_eq!(Some("value2".to_owned()), res);
    }
}
