use std::collections::VecDeque;
use std::fs::{File, remove_file};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use crate::active_func::{ActivateFunc, ActivateFunction};
use crate::matrix::Matrix;
use rand::prelude::*;
use crate::scanner::{Scanner, scanner_from_file, writer_to_file};


#[derive(Clone)]
pub struct DataNetwork {
    l: i32,
    pub size : Vec<i32>
}

impl DataNetwork {
    pub fn new() -> DataNetwork {
        DataNetwork {
            l: 0,
            size: vec![],
        }
    }

    pub fn from_data(len: i32, sz: Vec<i32>) -> DataNetwork {
        DataNetwork {
            l: len,
            size: sz.clone(),
        }
    }
}

#[derive(Clone)]
pub struct Network {
    act_func: ActivateFunction,
    l: i32,
    pub size : Vec<i32>,
    weights: Vec<Matrix>,
    bios: Vec<Vec<f64>>,
    neurons_val : Vec<Vec<f64>>,
    neurons_err : Vec<Vec<f64>>,
    neurons_bios_val : Vec<f64>,
}


impl Network {

    pub fn new() -> Network {
        let r = Network{
            act_func: ActivateFunction::new(),
            l: 0,
            size: Vec::new(),
            weights: Vec::new(),
            neurons_err: Vec::new(),
            neurons_val: Vec::new(),
            neurons_bios_val: Vec::new(),
            bios: Vec::new(),
        };
        return r.clone();
    }

    pub fn init <R: BufRead>(&mut self, data: DataNetwork, sc: &mut Scanner<R>) {
        self.act_func.set(sc);
        let mut rng = thread_rng();
        self.l = data.l;
        self.size = vec![0; self.l as usize];
        for i in 0..data.l {
            self.size[i as usize] = data.size[i as usize];
        }
        self.weights = vec![Matrix::new(); (self.l - 1) as usize];
        self.bios = vec![vec![];(self.l - 1) as usize];
        for i in 0..self.l - 1 {
            self.weights[i as usize].init(self.size[(i + 1) as usize], self.size[(i as usize)]);
            self.bios[i as usize] = vec![0.0; self.size[(i + 1) as usize] as usize];
            self.weights[i as usize].rand();
            for j in 0..self.size[(i + 1) as usize] {
                self.bios[i as usize][j as usize] = ((rng.next_u64() % 50) as f64) * 0.06 /
                    ((self.size[i as usize] + 15) as f64);
            }
        }
        self.neurons_val = vec![vec![]; self.l as usize];
        self.neurons_err = vec![vec![]; self.l as usize];
        for i in 0..self.l {
            self.neurons_val[i as usize] = vec![0.0;self.size[(i as usize)] as usize];
            self.neurons_err[i as usize] = vec![0.0;self.size[(i as usize)] as usize];
        }
        self.neurons_bios_val = vec![1.0; (self.l - 1) as usize];
    }

    pub fn print_config(&self) {
        println!("***********************************************************");
        print!("NetWork has {} layers\nSIZE[]: ", self.l);
        for i in 0..self.l {
            print!("{} ", self.size[i as usize]);
        }
        println!("\n***********************************************************\n");
    }

    pub fn set_input(&mut self, values: Vec<f64>) {
        for i in 0..self.size[0] {
            self.neurons_val[0 as usize][i as usize] = values[i as usize];
        }
    }

    pub fn search_max_index(&self, value : Vec<f64>) -> i32 {
        let mut mx = value[0];
        let mut prediction = 0;
        let mut tmp= 0.0;
        for i in 0..self.size[(self.l - 1) as usize] {
            tmp = value[i as usize];
            if tmp > mx {
                prediction = i;
                mx = tmp;
            }
        }
        return prediction;
    }

    pub fn forward_feed(& mut self) -> f64 {
        for i in 1..self.l {
            Matrix::multiply(self.weights[(i - 1) as usize].clone(),
                             (self.neurons_val[(i - 1) as usize]).clone(),
                             &mut self.neurons_val[i as usize]);
            Matrix::sum_vector(&mut self.neurons_val[i as usize],
                               self.bios[(i - 1) as usize].clone(), self.size[i as usize]);
            self.act_func.apply(&mut self.neurons_val[i as usize], self.size[i as usize]);
        }
        let pred = Network::search_max_index(self, self.neurons_val[(self.l - 1) as usize].clone());
        return pred as f64;
    }

    pub fn back_propogation(&mut self, expect: f64) {
        for i in 0..self.size[(self.l - 1) as usize] {
            if i != (expect as i32) {
                self.neurons_val[(self.l - 1) as usize][i as usize] =
                    -self.neurons_val[(self.l - 1) as usize][i as usize] * self.act_func.use_der(
                        self.neurons_val[(self.l - 1) as usize][i as usize]);
            } else {
                self.neurons_val[(self.l - 1) as usize][i as usize] =
                    (1.0 - self.neurons_val[(self.l - 1) as usize][i as usize]) *
                        self.act_func.use_der(self.neurons_val[(self.l - 1) as usize][i as usize]);
            }
        }
        for i in (1..self.l - 1).rev() {
            Matrix::multiply_transposed(self.weights[i as usize].clone(),
                             (self.neurons_val[(i + 1) as usize]).clone(),
                             &mut self.neurons_val[i as usize]);
            for j in 0..self.size[i as usize] {
                self.neurons_err[i as usize][j as usize] *= self.act_func.use_der(
                    self.neurons_val[i as usize][j as usize]);
            }
        }
    }

    pub fn update_weights(&mut self, mt : f64) {
        for i in 0..self.l - 1 {
            for j in 0..self.size[(i + 1) as usize] {
                for k in 0..self.size[i as usize] {
                    self.weights[i as usize][j as usize][k as usize] +=
                        self.neurons_val[i as usize][k as usize] *
                            self.neurons_err[(i + 1) as usize][j as usize] * mt;
                }
            }
        }
        for i in 0..self.l - 1 {
            for j in 0..self.size[(i + 1) as usize] {
                self.bios[i as usize][j as usize] +=
                    self.neurons_err[(i + 1) as usize][j as usize] * mt;
            }
        }
    }


    pub fn save_weights(&self) {
        let mut out = writer_to_file("weights.txt");
        for i in 0..self.l - 1 {
            self.weights[i as usize].write_to_output(& mut out);
        }
        for i in 0..self.l - 1{
            for j in 0..self.size[(i + 1) as usize] {
                write!(out, "{} ", self.bios[i as usize][j as usize]).expect("Failed to write bios");
            }
        }
        println!("Weights saved");
        //.expect("Error cn creating the file\n");
    }

    pub fn load_weights(&mut self) {
        let mut input = scanner_from_file("weights.txt");
        for i in 0..self.l - 1 {
            self.weights[i as usize].read_from_input(&mut input);
        }
        for i in 0..self.l - 1 {
            for j in 0..self.size[(i + 1) as usize] {
                self.bios[i as usize][j as usize] = input.next_value();
            }
        }
        println!("Weights loaded")
    }
}