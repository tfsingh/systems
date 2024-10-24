use anyhow::Result;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::{compaction, util};

pub fn compact(level: u64, data_file: &Path) -> Result<()> {
    let file_name = data_file.file_name().unwrap().to_string_lossy();

    let level_file = if level == 0 {
        data_file.into()
    } else {
        data_file.with_file_name(format!("{}_{}", file_name, level))
    };

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&level_file)?;

    let mut buffer = [0; 8];
    file.read_exact(&mut buffer)?;
    let offset = u64::from_le_bytes(buffer);

    let mut first_sst = vec![0; (offset - 8) as usize];
    file.seek(SeekFrom::Start(8))?;
    file.read_exact(&mut first_sst)?;

    let mut second_sst = Vec::new();
    file.seek(SeekFrom::Start(offset))?;
    file.read_to_end(&mut second_sst)?;

    let mut map = BTreeMap::new();

    for (key, value) in util::read_pairs_from_buffer(&first_sst)? {
        map.insert(key, value);
    }
    for (key, value) in util::read_pairs_from_buffer(&second_sst)? {
        map.insert(key, value);
    }

    let next_level_file = data_file.with_file_name(format!("{}_{}", file_name, level + 1));
    if next_level_file.exists() {
        util::write_map_to_existing_file(&map, &next_level_file)?;
        compaction::compact(level + 1, data_file)?;
    } else {
        util::write_map_to_new_file(&map, &next_level_file)?;
    }
    std::fs::remove_file(level_file)?;

    Ok(())
}
