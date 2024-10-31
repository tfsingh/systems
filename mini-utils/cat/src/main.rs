use std::{
    env, fs,
    io::{self, Write},
};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    for arg in args {
        let data: Vec<u8> = fs::read(arg).unwrap();
        io::stdout().write_all(&data).unwrap();
        io::stdout().flush().unwrap();
    }
}
