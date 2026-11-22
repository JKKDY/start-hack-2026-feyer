#![allow(unused_variables, unused_mut)]
include!("formulas.rs");
use std::slice;

// Per component: function, swept-input bitmask, write start/end, seven parameter indices.
// Per write: component output lane and its destination in the coefficient array.
#[no_mangle]
pub unsafe extern "C" fn run_coefficients(
    backward: bool,
    batch: usize,
    width: usize,
    count: usize,
    nout: usize,
    x: *const C,
    plan: *const i32,
    writes: *const i32,
    cot: *const C,
    out: *mut C,
) {
    let x = slice::from_raw_parts(x, batch * width);
    let plan = slice::from_raw_parts(plan, count * 11);
    let writes = slice::from_raw_parts(writes, nout * 2);
    let cot = if backward {
        slice::from_raw_parts(cot, batch * nout)
    } else {
        &[]
    };
    let out = slice::from_raw_parts_mut(
        out,
        if backward {
            batch * width
        } else {
            batch * nout
        },
    );
    out.fill(C::default());
    for row in plan.chunks_exact(11) {
        macro_rules! run {
            ($m:expr) => {
                process::<$m>(row, batch, width, nout, x, writes, cot, out, backward)
            };
        }
        match row[1] {
            0 => run!(0),
            1 => run!(1),
            2 => run!(2),
            4 => run!(4),
            8 => run!(8),
            16 => run!(16),
            32 => run!(32),
            64 => run!(64),
            _ => run!(127),
        }
    }
}
fn process<const M: u8>(
    row: &[i32],
    batch: usize,
    width: usize,
    nout: usize,
    x: &[C],
    writes: &[i32],
    cot: &[C],
    out: &mut [C],
    backward: bool,
) {
    let invariant = M == 0;
    let first = std::array::from_fn(|j| x[row[4 + j] as usize]);
    let mut cache = [C::default(); 128];
    pre::<M>(row[0], &first, &mut cache);
    let destinations = &writes[(row[2] as usize) * 2..(row[3] as usize) * 2];
    for b in 0..if invariant { 1 } else { batch } {
        let p = std::array::from_fn(|j| x[b * width + row[j + 4] as usize]);
        let stop = if invariant { batch } else { b + 1 };
        if backward {
            let mut g = [C::default(); 4];
            for f in b..stop {
                for w in destinations.chunks_exact(2) {
                    g[w[0] as usize] += cot[f * nout + w[1] as usize];
                }
            }
            let dp = back::<M>(row[0], &p, &g, &cache);
            for j in 0..7 {
                out[b * width + row[4 + j] as usize] += dp[j];
            }
        } else {
            let v = eval::<M>(row[0], &p, &cache);
            for f in b..stop {
                for w in destinations.chunks_exact(2) {
                    out[f * nout + w[1] as usize] = v[w[0] as usize];
                }
            }
        }
    }
}
