use std::{env, io::{self, Write}, fs};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let pattern = args.get(0).expect("Pattern not supplied");

    for file in &args[1..] {
        let data: String = fs::read_to_string(file).unwrap();

        let occurences = find_occurences_of_pattern(&pattern, &data);

        for occurence in occurences {
            io::stdout().write(occurence.as_bytes()).unwrap();
        }

        io::stdout().flush().unwrap();
    }
}

fn find_occurences_of_pattern(pattern: &str, data: &str) -> Vec<String> {
    let lines = data.split("\n");
    let mut result = vec![];

    for (i, line) in lines.enumerate() {
        if line.contains(pattern) {
            result.push(format!("{}: {}\n", i + 1, line.trim()));
        }
    }

    result
}
