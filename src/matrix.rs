use std::collections::VecDeque;
use std::fs::{File, read};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::ops::{Index, IndexMut};
use std::ptr::write;
use std::vec::Vec;
use rand::prelude::*;
use crate::scanner::Scanner;

#[derive(Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rws: i32,
    pub cms: i32,
}


impl Matrix {
    pub fn init(&mut self, r: i32, c: i32) {
        self.rws = r;
        self.cms = c;
        self.data = vec![vec![0.; c as usize]; r as usize];
    }

    pub fn new() -> Matrix {
        let m = Matrix{
            rws: 0,
            cms: 0,
            data: Vec::new()
        };
        return m.clone();
    }

    pub fn rand(&mut self) {
        let mut rng = thread_rng();
        for i in 0..self.rws {
            for j in 0..self.cms {
                self.data[i as usize][j as usize] = ((rng.next_u64() % 1000) as f64) / 1000_f64;
            }
        }
    }

    pub fn transpose(m: Matrix) -> Matrix {
        let mut ans: Matrix = Matrix {
            data: vec![],
            rws: m.cms,
            cms: m.rws,
        };
        ans.init(m.cms, m.rws);
        for i in 0..ans.cms {
            for j in 0..ans.rws {
                ans.data[j as usize][i as usize] = m.data[i as usize][j as usize];
            }
        }
        return ans;
    }

    pub fn multiply(m1: Matrix, neuron: Vec<f64>, c: &mut [f64]) {
        for i in 0..m1.rws {
            let mut tmp = 0.0;
            for j in 0..m1.cms {
                tmp += m1.data[i as usize][j as usize] * neuron[j as usize];
            }
            c[i as usize] = tmp;
        }
    }

    pub fn multiply_transposed(m1: Matrix, neuron: Vec<f64>, c: &mut [f64]) {
        for i in 0..m1.cms {
            let mut tmp = 0.0;
            for j in 0..m1.rws {
                tmp += m1.data[j as usize][i as usize] * neuron[j as usize];
            }
            c[i as usize] = tmp;
        }
    }

    pub fn sum_vector(a: &mut [f64], b: Vec<f64>, n: i32)
    {
        for i in 0..n {
            a[i as usize] += b[i as usize];
        }
    }
    
    pub fn write_to_output<W: Write>(&self, out: &mut BufWriter<W>) {
        for i in 0..self.rws {
            for j in 0..self.cms {
                write!(out, "{} ", self.data[i as usize][j as usize]).expect("Filed to write matrix");
            }
        }
    }

    pub fn read_from_input <R: BufRead>(&mut self, sc: &mut Scanner<R>) {
        for i in 0..self.rws {
            for j in 0..self.cms {
                self.data[i as usize][j as usize] = sc.next_value();
            }
        }
    }
}


impl Index<usize> for Matrix {
    type Output = Vec<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.data[index];
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Vec<f64> {
        return &mut self.data[index];
    }
}

// impl Clone for Matrix {
//     fn clone(&self) -> Self {
//         return Matrix {
//             data: self.data.clone(),
//             rws: self.rws,
//             cms: self.cms,
//         };
//     }
// }
