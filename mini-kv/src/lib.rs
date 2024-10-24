use anyhow::{anyhow, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

mod compaction;
mod sst;
mod util;
mod wal;

use sst::SST;
use wal::WAL;

pub struct Cyrus {
    sst: SST,
    wal: WAL,
    memtable: BTreeMap<String, String>,
    memtable_size: u64,
}

const CYRUS_TOMB: &str = "_cyrus-tomb_";

impl Cyrus {
    pub fn new(name: &str) -> Self {
        let name = "dbs/".to_owned() + name;
        let dir_path = Path::new(&name);
        let data_file = dir_path.join("data");
        let util_file = dir_path.join("util");

        if let Err(_) = fs::create_dir_all(dir_path) {
            panic!("Couldn't create database");
        }

        if !util_file.exists() {
            if let Err(_) = fs::File::create(&util_file) {
                panic!("Couldn't create util file");
            }
        }

        let sst = SST::new(data_file.into());
        let mut wal = WAL::new(util_file.into());

        let mut memtable = BTreeMap::new();
        let prev_writes = wal.restore().expect("Unable to restore write ahead log");
        prev_writes.iter().for_each(|(k, v)| {
            memtable.insert(k.to_owned(), v.to_owned());
        });

        Cyrus {
            sst,
            wal,
            memtable,
            memtable_size: 0,
        }
    }

    pub fn insert(&mut self, key: &str, value: &str) -> Result<()> {
        if key == "" || value == "" || key.len() > 65_000 || value.len() > 65_000 {
            return Err(anyhow!("Invalid key or value"));
        }
        self.write_internal(key, value)?;
        Ok(())
    }

    pub fn remove(&mut self, key: &str) -> Result<()> {
        if key == "" || key.len() > 65_000 {
            return Err(anyhow!("Invalid key or value"));
        }
        self.write_internal(key, CYRUS_TOMB)?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let potential_result = self.memtable.get(key).cloned();
        if let Some(res) = &potential_result {
            if res == CYRUS_TOMB {
                None
            } else {
                potential_result
            }
        } else {
            let result = self.sst.get(key);
            match result {
                Some(res) if res == CYRUS_TOMB => None,
                Some(res) => Some(res),
                None => None,
            }
        }
    }

    fn write_internal(&mut self, key: &str, value: &str) -> Result<()> {
        self.memtable.insert(key.to_owned(), value.to_owned());
        // self.wal
        //     .write(key, value)
        //     .expect("Failure while persisting write");
        self.memtable_size += (key.len() + value.len()) as u64;
        if self.memtable_size > util::MAX_MEMTABLE_SIZE.try_into()? {
            if self.sst.flush(&self.memtable).is_ok() {
                self.wal.clear()?;
                self.memtable.clear();
                self.memtable_size = 0;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn test(name: &str) -> Self {
        let new_name = "dbs/".to_owned() + name;
        let dir_path = Path::new(&new_name);
        let _ = fs::remove_dir_all(dir_path);
        Cyrus::new(name)
    }

    #[cfg(test)]
    pub fn get_wal(&mut self) -> &mut WAL {
        &mut self.wal
    }

    #[cfg(test)]
    pub fn get_sst(&mut self) -> &mut SST {
        &mut self.sst
    }

    #[cfg(test)]
    pub fn get_memtable(&mut self) -> &mut BTreeMap<String, String> {
        &mut self.memtable
    }

    #[cfg(test)]
    pub fn clear_mem(&mut self) {
        self.memtable.clear();
    }
}
