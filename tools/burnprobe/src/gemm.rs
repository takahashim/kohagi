//! GEMM だけを同じ形状で candle と burn で比較する。
//! ruri-v3-130m の MLP 射影: [rows*seq, 512] @ [512, 4096]
use std::time::Instant;

fn main() {
    let (m, k, n) = (4 * 454usize, 512usize, 4096usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 89) as f32 * 0.01 - 0.5).collect();
    let reps = 10;
    let flops = 2.0 * m as f64 * k as f64 * n as f64 * reps as f64;

    // --- candle
    {
        use candle_core::{Device, Tensor};
        let d = Device::Cpu;
        let ta = Tensor::from_vec(a.clone(), (m, k), &d).unwrap();
        let tb = Tensor::from_vec(b.clone(), (k, n), &d).unwrap();
        let _ = ta.matmul(&tb).unwrap(); // warm
        let t = Instant::now();
        for _ in 0..reps {
            let c = ta.matmul(&tb).unwrap();
            std::hint::black_box(c.sum_all().unwrap().to_scalar::<f32>().unwrap());
        }
        let e = t.elapsed();
        println!("candle  : {:?}  {:.1} GFLOP/s", e, flops / e.as_secs_f64() / 1e9);
    }

    // --- burn flex
    {
        use burn::tensor::{Tensor, TensorData};
        type B = burn::backend::Flex;
        let d = Default::default();
        let ta = Tensor::<B, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<B, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let _ = ta.clone().matmul(tb.clone()).sum().into_scalar();
        let t = Instant::now();
        for _ in 0..reps {
            let c = ta.clone().matmul(tb.clone());
            std::hint::black_box(c.sum().into_scalar());
        }
        let e = t.elapsed();
        println!("burn flex: {:?}  {:.1} GFLOP/s", e, flops / e.as_secs_f64() / 1e9);
    }

    // --- burn ndarray
    {
        use burn::tensor::{Tensor, TensorData};
        type B = burn::backend::NdArray;
        let d = Default::default();
        let ta = Tensor::<B, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<B, 2>::from_data(TensorData::new(b, [k, n]), &d);
        let _ = ta.clone().matmul(tb.clone()).sum().into_scalar();
        let t = Instant::now();
        for _ in 0..reps {
            let c = ta.clone().matmul(tb.clone());
            std::hint::black_box(c.sum().into_scalar());
        }
        let e = t.elapsed();
        println!("burn ndarray: {:?}  {:.1} GFLOP/s", e, flops / e.as_secs_f64() / 1e9);
    }
}
