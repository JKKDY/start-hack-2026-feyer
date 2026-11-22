use num_complex::Complex64 as C;
use std::{ffi::c_void, slice};

extern "C" {
    fn optical_prepare(
        n: i32,
        nz: i32,
        data: *const C,
        rows: *const i32,
        ptr: *const i32,
    ) -> *mut c_void;
    fn optical_finish(plan: *mut c_void);
    fn optical_factor(
        n: i32,
        nz: i32,
        data: *const C,
        rows: *const i32,
        ptr: *const i32,
        plan: *mut c_void,
    ) -> *mut c_void;
    fn optical_solve(f: *mut c_void, mode: i32, nrhs: i32, x: *mut C) -> i32;
    fn optical_release(f: *mut c_void);
}
struct Plan(*mut c_void);
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe { optical_finish(self.0) }
    }
}
struct Factors(*mut c_void);
impl Drop for Factors {
    fn drop(&mut self) {
        unsafe {
            optical_release(self.0);
        }
    }
}

struct Matrix<'a> {
    values: &'a [C],
    rows: &'a [i32],
    ptr: &'a [i32],
}
struct Work {
    rhs: Vec<C>,
    x: Vec<C>,
    residual: Vec<C>,
}
impl Work {
    fn new(size: usize) -> Self {
        Self {
            rhs: vec![C::default(); size],
            x: vec![C::default(); size],
            residual: vec![C::default(); size],
        }
    }
    fn refined(
        &mut self,
        factor: &Factors,
        a: &Matrix,
        input: &[C],
        output: &mut [C],
        count: usize,
        trans: bool,
        conjugate: bool,
        steps: usize,
    ) -> bool {
        let n = a.ptr.len() - 1;
        let len = n * count;
        let (rhs, x, res) = (
            &mut self.rhs[..len],
            &mut self.x[..len],
            &mut self.residual[..len],
        );
        for j in 0..count {
            for i in 0..n {
                rhs[j * n + i] = input[i * count + j];
            }
        }
        let solve = |v: &mut [C]| {
            // SuperLU provides A, A^T, A^H. Conjugate the RHS and solution for conj(A).
            if conjugate && !trans {
                for z in v.iter_mut() {
                    *z = z.conj();
                }
            }
            let mode = if !trans {
                0
            } else if conjugate {
                2
            } else {
                1
            };
            let ok = unsafe { optical_solve(factor.0, mode, count as i32, v.as_mut_ptr()) } == 0;
            if conjugate && !trans {
                for z in v.iter_mut() {
                    *z = z.conj();
                }
            }
            ok
        };
        x.copy_from_slice(rhs);
        if !solve(x) {
            return false;
        }
        for _ in 0..steps {
            res.fill(C::default());
            for j in 0..count {
                let off = j * n;
                for col in 0..n {
                    for k in a.ptr[col] as usize..a.ptr[col + 1] as usize {
                        let row = a.rows[k] as usize;
                        let v = if conjugate {
                            a.values[k].conj()
                        } else {
                            a.values[k]
                        };
                        if trans {
                            res[off + col] += v * x[off + row];
                        } else {
                            res[off + row] += v * x[off + col];
                        }
                    }
                }
            }
            for i in 0..len {
                res[i] = rhs[i] - res[i];
            }
            if !solve(res) {
                return false;
            }
            for i in 0..len {
                x[i] += res[i];
            }
        }
        for j in 0..count {
            for i in 0..n {
                output[i * count + j] = x[j * n + i];
            }
        }
        true
    }
}

// Caller provides contiguous complex128 buffers with the shapes checked in Python.
#[no_mangle]
pub unsafe extern "C" fn optical_sweep(
    batch: usize,
    n: usize,
    nz: usize,
    det: usize,
    steps: usize,
    trans: bool,
    values: *const C,
    rows: *const i32,
    ptr: *const i32,
    signal: *const C,
    detector: *const C,
    out_signal: *mut C,
    out_detector: *mut C,
) -> i32 {
    if n == 0
        || nz == 0
        || n > i32::MAX as usize
        || nz > i32::MAX as usize
        || det > i32::MAX as usize
    {
        return -1;
    }
    let rows = slice::from_raw_parts(rows, nz);
    let ptr = slice::from_raw_parts(ptr, n + 1);
    if ptr[0] != 0
        || ptr[n] != nz as i32
        || ptr.windows(2).any(|w| w[0] > w[1])
        || rows.iter().any(|&r| r < 0 || r as usize >= n)
    {
        return -1;
    }
    // Noncanonical CSC matrices go through SciPy's duplicate consolidation.
    if ptr.windows(2).any(|w| {
        rows[w[0] as usize..w[1] as usize]
            .windows(2)
            .any(|r| r[0] >= r[1])
    }) {
        return -1;
    }
    if batch == 0 {
        return 0;
    }
    let plan = Plan(optical_prepare(
        n as i32,
        nz as i32,
        values,
        rows.as_ptr(),
        ptr.as_ptr(),
    ));
    if plan.0.is_null() {
        return 1;
    }
    let mut work = Work::new(n * det.max(1));
    for b in 0..batch {
        let a = Matrix {
            values: slice::from_raw_parts(values.add(b * nz), nz),
            rows,
            ptr,
        };
        let raw = optical_factor(
            n as i32,
            nz as i32,
            a.values.as_ptr(),
            rows.as_ptr(),
            ptr.as_ptr(),
            plan.0,
        );
        if raw.is_null() {
            return 1;
        }
        let f = Factors(raw);
        if !work.refined(
            &f,
            &a,
            slice::from_raw_parts(signal.add(b * n), n),
            slice::from_raw_parts_mut(out_signal.add(b * n), n),
            1,
            trans,
            false,
            steps,
        ) {
            return 1;
        }
        if det > 0
            && !work.refined(
                &f,
                &a,
                slice::from_raw_parts(detector.add(b * n * det), n * det),
                slice::from_raw_parts_mut(out_detector.add(b * n * det), n * det),
                det,
                !trans,
                true,
                steps,
            )
        {
            return 1;
        }
    }
    0
}
