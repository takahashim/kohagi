//! burn-flex で融合カーネルが書けるか、書いたら効くか。
//!
//! 対象は RoPE: kohagi の candle 経路は candle_nn::rotary_emb::rope という
//! 融合カーネル1回で済ませ、私の Burn 実装は narrow/neg/cat/mul/add で組んでいる。
//! 1フォワードあたり 19層 x (q,k) = 38 回走る。

use std::time::Instant;

use burn::backend::flex::FlexTensor;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};

type B = burn::backend::Flex;

const ROWS: usize = 4;
const HEADS: usize = 8;
const SEQ: usize = 454;
const HD: usize = 64;
const CALLS: usize = 38;

fn composed(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let half = HD / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let rotated = Tensor::cat(vec![-x2, x1], 3);
    x * cos + rotated * sin
}

/// 同じ計算を生スライス1パスで。`storage()` はゼロコピービュー。
fn fused(x: Tensor<B, 4>, cos: &[f32], sin: &[f32]) -> Tensor<B, 4> {
    let prim: FlexTensor = x.into_primitive().tensor();
    let src: &[f32] = prim.storage::<f32>();
    let half = HD / 2;
    let mut out = vec![0f32; src.len()];
    // [rows, heads, seq, hd] を行ごとに舐める。cos/sin は [1,1,seq,hd]。
    for r in 0..ROWS * HEADS {
        for p in 0..SEQ {
            let o = (r * SEQ + p) * HD;
            let t = p * HD;
            for i in 0..half {
                let (a, b) = (src[o + i], src[o + i + half]);
                out[o + i] = a * cos[t + i] - b * sin[t + i];
                out[o + i + half] = b * cos[t + i + half] + a * sin[t + i + half];
            }
        }
    }
    let d = Default::default();
    Tensor::from_primitive(TensorPrimitive::Float(FlexTensor::from_data(
        TensorData::new(out, [ROWS, HEADS, SEQ, HD]),
    )))
    .to_device(&d)
}

fn main() {
    let d = Default::default();
    let n = ROWS * HEADS * SEQ * HD;
    let xs: Vec<f32> = (0..n).map(|i| (i % 101) as f32 * 0.01 - 0.5).collect();
    let cs: Vec<f32> = (0..SEQ * HD).map(|i| ((i % 71) as f32 * 0.02).cos()).collect();
    let sn: Vec<f32> = (0..SEQ * HD).map(|i| ((i % 71) as f32 * 0.02).sin()).collect();

    let x = Tensor::<B, 4>::from_data(TensorData::new(xs.clone(), [ROWS, HEADS, SEQ, HD]), &d);
    let cos = Tensor::<B, 4>::from_data(TensorData::new(cs.clone(), [1, 1, SEQ, HD]), &d);
    let sin = Tensor::<B, 4>::from_data(TensorData::new(sn.clone(), [1, 1, SEQ, HD]), &d);

    // 一致確認
    let a: Vec<f32> = composed(x.clone(), cos.clone(), sin.clone())
        .into_data().convert::<f32>().to_vec().unwrap();
    let b: Vec<f32> = fused(x.clone(), &cs, &sn).into_data().convert::<f32>().to_vec().unwrap();
    let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
    println!("一致: max|diff| = {worst:.3e}");

    let t = Instant::now();
    for _ in 0..CALLS {
        let r = composed(x.clone(), cos.clone(), sin.clone());
        std::hint::black_box(r.into_primitive());
    }
    let e1 = t.elapsed();

    let t = Instant::now();
    for _ in 0..CALLS {
        let r = fused(x.clone(), &cs, &sn);
        std::hint::black_box(r.into_primitive());
    }
    let e2 = t.elapsed();

    println!("composed (narrow/neg/cat/mul/add) x{CALLS} : {e1:?}");
    println!("fused    (生スライス1パス)        x{CALLS} : {e2:?}   {:.2}x", e1.as_secs_f64() / e2.as_secs_f64());
}
