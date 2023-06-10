use std::io::BufRead;
use crate::scanner::Scanner;

#[derive(Clone)]
pub enum ActivateFunc {
    Sigmoid = 1,
    ReLU,
    Tnx,
}

#[derive(Clone)]
pub struct ActivateFunction {
    pub active_func: ActivateFunc,
}


impl ActivateFunction {
    pub fn new() -> ActivateFunction {
        let r = ActivateFunction {
            active_func : ActivateFunc::Sigmoid
        };
        return r.clone();
    }
    pub fn set <R: BufRead>(&mut self, s : &mut Scanner<R>) {
        println!("Set actFunc pls:\n1 - sigmoid\n2 - ReLU\n3 - th(x)");
        let input: i32 = s.next_value();
        match input {
            1 => self.active_func = ActivateFunc::Sigmoid,
            2 => self.active_func = ActivateFunc::ReLU,
            3 => self.active_func = ActivateFunc::Tnx,
            _ => panic!("Error read actFunc"),
        }
    }

    pub fn apply(&self, val: &mut [f64], n: i32) {
        match self.active_func {
            ActivateFunc::Sigmoid => {
                for i in 0..n {
                    val[i as usize] = 1.0 / (1.0 + (-val[i as usize]).exp());
                }
            }
            ActivateFunc::ReLU => {
                for i in 0..n {
                    if val[i as usize] < 0 as f64 {
                        val[i as usize] *= 0.01;
                    } else if val[i as usize] > 1.0 {
                        val[i as usize] = 1.0 + 0.01 * (val[i as usize] - 1.0);
                    }
                }
            }
            ActivateFunc::Tnx => {
                for i in 0..n {
                    if val[i as usize] < 0 as f64 {
                        val[i as usize] = 0.01 * (val[i as usize].exp() -
                            (-val[i as usize]).exp()) / (val[i as usize].exp() +
                            (-val[i as usize]).exp());
                    } else {
                        val[i as usize] = (val[i as usize].exp() -
                            (-val[i as usize]).exp()) / (val[i as usize].exp() +
                            (-val[i as usize]).exp());
                    }
                }
            }
        }
    }

    pub fn use_der(&self, mut val: f64) -> f64 {
        match self.active_func {
            ActivateFunc::Sigmoid => {
                val = 1.0 * (1.0 + (-val).exp());
            }
            ActivateFunc::ReLU => {
                if val < 0.0 || val > 1.0 {
                    val = 0.01;
                }
            }
            ActivateFunc::Tnx => {
                if val < 0.0 {
                    val = 0.01 * (val.exp() -
                        (-val).exp()) / (val.exp() +
                        (-val).exp());
                } else {
                    val = (val.exp() -
                        (-val).exp()) / (val.exp() +
                        (-val).exp());
                }
            }
        }
        return val;
    }
}