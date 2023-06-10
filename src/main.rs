mod matrix;
use std::io;
use std::fs::File;
use std::io::{BufRead, BufReader, Stdin};
use std::path::Path;
use std::time::{Duration, Instant};
use matrix::Matrix;
mod active_func;
use active_func::ActivateFunc;
use active_func::ActivateFunction;
mod network;
mod scanner;
use network::DataNetwork;
use network::Network;
use crate::scanner::{Scanner, scanner_from_file, scanner_from_stdin};

#[derive(Clone)]
pub struct DataInfo {
    pixels: Vec<f64>,
    digit: i32
}
impl DataInfo {
    pub fn new() -> DataInfo {
        DataInfo {
            pixels: Vec::new(),
            digit: 0
        }
    }

    pub fn from_data(p: Vec<f64>, d: i32) -> DataInfo {
        DataInfo {
            pixels: p.clone(),
            digit: d
        }
    }
}

pub fn read_data_network(path: &str) -> DataNetwork {
    let mut sc = scanner_from_file(path);
    let cnt: i32 = sc.next_value();
    let mut numbers = Vec::new();
    for _ in 0..cnt {
        let v : i32 = sc.next_value();
        numbers.push(v);
    }
    let mut data = DataNetwork::from_data(numbers.len() as i32, numbers);
    return data;
}

pub fn read_data(path: &str, sz: i32, ex: &mut i32) -> Vec<DataInfo> {
    let mut sc = scanner_from_file(path);
    let mut cnt = sc.next_value();
    println!("{} {}", cnt, sz);
    *ex = cnt;
    let mut d = vec![DataInfo::new(); cnt as usize];
    println!("Examples: {}", cnt);
    for i in 0..cnt {
        d[i as usize].digit = sc.next_value();
        for _ in 0..sz {
            d[i as usize].pixels.push(sc.next_value());
        }
    }
    println!("Loading...");
    d
}

fn main() {
    let mut nw = Network::new();
    let mut nw_config = DataNetwork::new();
    let mut data: Vec<DataInfo>;
    let mut ra = 0.;
    let mut right: f64;
    let mut predict: f64;
    let mut maxra = 0.;
    let mut epoch = 0;
    let mut study = 0;
    let mut repeat = 1;
    let mut time: Duration;
    let mut scan = scanner_from_stdin();
    nw_config = read_data_network("config.txt");
    nw.init(nw_config.clone(), & mut scan);
    nw.print_config();
    while repeat == 1 {
        println!("STUDY? (1/0)");
        study = scan.next_value();
        if study == 1 {
            let mut examples : i32 = 0;
            data = read_data("lib_MNIST_edit.txt", nw_config.clone().size[0], &mut examples);
            let begin = Instant::now();
            while ra / (examples as f64) * 100.0 < 100.0 && epoch < 20 {
                ra = 0.0;
                let t1 = Instant::now();
                for i in 0..examples {
                    nw.set_input(data[i as usize].pixels.clone());
                    right = data[i as usize].digit as f64;
                    predict = nw.forward_feed();
                    if predict != right {
                        nw.back_propogation(right);
                        nw.update_weights(0.15 * (-(epoch as f64) / 20.0).exp());
                    }
                    else {
                        ra += 1.0;
                    }
                }
                let t2 = Instant::now();
                time = t2.duration_since(t1);
                if ra > maxra { maxra = ra;}
                println!("ra: {:.2}%\t maxra: {:.2}%\t epoch: {}\t TIME: {:.2}s", ra / (examples as f64) * 100.0, maxra / (examples as f64) * 100.0, epoch, time.as_secs_f64());
                epoch += 1;
            }
            let end = Instant::now();
            time = end.duration_since(begin);
            println!("TIME: {:.2} min", time.as_secs_f64() / 60.0);
            nw.save_weights();
        }
        else {
            nw.load_weights();
        }
        println!("Test?(1/0)");
        let mut to_start_test: i32 = scan.next_value();
        if to_start_test == 1 {
            let mut ex_tests = 0;
            let mut rim = String::new();
            let mut data_test: Vec<DataInfo>;
            data_test = read_data("test.txt", nw_config.clone().size[0], &mut ex_tests);
            ra = 0.0;
            for i in 0..ex_tests {
                nw.set_input(data_test[i as usize].pixels.clone());
                predict = nw.forward_feed();
                right = data_test[i as usize].digit as f64;
                if right == predict {
                    ra += 1.0;
                }
                let r1: String;
                match predict as i32 {
                    0 => r1 = "0".to_string(),
                    1 => r1 = "1".to_string(),
                    2 => r1 = "2".to_string(),
                    3 => r1 = "3".to_string(),
                    4 => r1 = "4".to_string(),
                    5 => r1 = "5".to_string(),
                    6 => r1 = "6".to_string(),
                    7 => r1 = "7".to_string(),
                    8 => r1 = "8".to_string(),
                    _ => r1 = "9".to_string(),
                }
                println!("Your digit {} = {}", i + 1, r1);
                rim.push_str(&r1);
            }
            println!("RA: {:.2}%", ra / (ex_tests as f64) * 100.0);

            //////////////////////////////////////////////////////////
            println!("Полученное число: {}", rim);
            //////////////////////////////////////////////////////////
        }
        println!("Repeat? (1/0)");
        repeat = scan.next_value();
    }
}
