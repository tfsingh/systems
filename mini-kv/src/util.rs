use anyhow::{anyhow, Result};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub const MAX_MEMTABLE_SIZE: u64 = 1_000_000;

pub fn read_pairs_from_buffer(buffer: &[u8]) -> Result<Vec<(String, String)>> {
    let mut result = vec![];
    let mut i = 0;
    while i < buffer.len() {
        if buffer.len() < i + 4 {
            return Err(anyhow!("Unexpected end of file"));
        }

        let (key_len, value_len) = {
            let key_len = u16::from_le_bytes(buffer[i..i + 2].try_into()?);
            let value_len = u16::from_le_bytes(buffer[i + 2..i + 4].try_into()?);
            i += 4;
            (key_len as usize, value_len as usize)
        };

        if buffer.len() < i + key_len + value_len {
            return Err(anyhow!("Unexpected end of file"));
        }

        let key = String::from_utf8(buffer[i..(i + key_len)].to_vec())?;
        let value = String::from_utf8(buffer[(i + key_len)..(i + key_len + value_len)].to_vec())?;
        i += key_len + value_len;
        if key == "" || value == "" {
            break;
        }
        result.push((key, value));
    }

    Ok(result)
}

pub fn write_pair_to_file(key: &str, value: &str, file: &mut File) -> Result<()> {
    let key_len = key.len() as u16;
    let value_len = value.len() as u16;

    file.write_all(&key_len.to_le_bytes())?;
    file.write_all(&value_len.to_le_bytes())?;
    file.write_all(key.as_bytes())?;
    file.write_all(value.as_bytes())?;
    Ok(())
}

pub fn write_map_to_new_file(map: &BTreeMap<String, String>, new_file: &Path) -> Result<()> {
    let mut new_file = OpenOptions::new().write(true).create(true).open(new_file)?;

    new_file.write_all(&[0; 8])?;
    let mut offset = 8u64;

    new_file.seek(SeekFrom::Start(8))?;
    for (key, value) in map.iter() {
        write_pair_to_file(key, value, &mut new_file)?;
        offset += (4 + key.len() + value.len()) as u64;
    }

    new_file.seek(SeekFrom::Start(0))?;
    new_file.write_all(&offset.to_le_bytes())?;
    Ok(())
}

pub fn write_map_to_existing_file(map: &BTreeMap<String, String>, file: &Path) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .open(file)?;

    let offset = get_offset(&mut file)?;

    file.seek(SeekFrom::Start(offset))?;
    for (key, value) in map.iter() {
        write_pair_to_file(key, value, &mut file)?;
    }
    Ok(())
}

pub fn read_recent_sst(file: &PathBuf) -> Result<Vec<u8>> {
    let mut file = OpenOptions::new().read(true).open(file)?;
    let offset = get_offset(&mut file)?;

    file.seek(SeekFrom::Start(offset))?;
    let mut buffer = vec![];

    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

pub fn read_older_sst(file: &PathBuf) -> Result<Vec<u8>> {
    let mut file = OpenOptions::new().read(true).open(file)?;
    let offset = get_offset(&mut file)?;

    file.seek(SeekFrom::Start(8)).unwrap();
    let mut buffer = vec![0; (offset - 8) as usize];
    file.read_exact(&mut buffer).unwrap();
    Ok(buffer)
}

fn get_offset(file: &mut File) -> Result<u64> {
    let mut offset_bytes = [0u8; 8];
    file.read_exact(&mut offset_bytes)?;
    Ok(u64::from_le_bytes(offset_bytes))
}
