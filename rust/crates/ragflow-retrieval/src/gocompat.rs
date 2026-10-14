//! Go runtime behaviour the retrieval path depends on, reproduced exactly:
//! `fmt` float verbs, `strings.ToLower`, `strings.Fields`, numpy-style
//! pairwise summation, glibc `log10`/`sqrt`, and `sort.Slice` (pdqsort,
//! unstable: equal elements land where Go puts them).

use std::cmp::Ordering;

extern "C" {
    #[link_name = "log10"]
    fn c_log10(x: f64) -> f64;
    #[link_name = "sqrt"]
    fn c_sqrt(x: f64) -> f64;
}

/// `common.PyLog10`: the C library's `log10` (cgo build).
pub fn py_log10(x: f64) -> f64 {
    // SAFETY: pure libm function.
    unsafe { c_log10(x) }
}

/// `common.PySqrt`: the C library's `sqrt`.
pub fn py_sqrt(x: f64) -> f64 {
    // SAFETY: pure libm function.
    unsafe { c_sqrt(x) }
}

/// `common.PairwiseSum` (numpy's algorithm). Consumes a copy.
pub fn pairwise_sum(xs: &[f64]) -> f64 {
    let mut xs = xs.to_vec();
    let mut n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n < 16 {
        let mut sum = 0.0;
        for x in &xs {
            sum += x;
        }
        return sum;
    }
    while n > 1 {
        let m = n / 2;
        for i in 0..m {
            xs[i] = xs[2 * i] + xs[2 * i + 1];
        }
        if n % 2 != 0 {
            xs[m] = xs[n - 1];
            n = m + 1;
        } else {
            n = m;
        }
    }
    xs[0]
}

/// Go `strings.ToLower`: per-rune simple lowercase mapping.
pub fn to_lower(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c.to_ascii_lowercase());
        } else {
            // Only U+0130 lowercases to more than one char; Go maps it to 'i',
            // the first char of the full mapping.
            out.push(c.to_lowercase().next().unwrap_or(c));
        }
    }
    out
}

/// Go `strings.Fields` (unicode.IsSpace == Unicode White_Space).
pub fn fields(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

/// Rune count (`utf8.RuneCountInString`, `len([]rune(s))`).
pub fn rune_len(s: &str) -> usize {
    s.chars().count()
}

/// Shortest round-trip decimal digits and decimal-point position of a finite,
/// non-zero `x` (digits without sign, `dp` as in Go's `decimalSlice.dp`).
fn shortest_digits(x: f64) -> (String, i32) {
    let e = format!("{:e}", x.abs());
    let (mant, exp) = e.split_once('e').expect("LowerExp has an exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let exp: i32 = exp.parse().expect("exponent");
    (digits, exp + 1)
}

/// Go `%v` / `strconv.FormatFloat(x, 'g', -1, 64)`.
pub fn fmt_v(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "+Inf".into()
        } else {
            "-Inf".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let neg = x < 0.0;
    let (d, dp) = shortest_digits(x);
    let nd = d.len() as i32;
    let exp = dp - 1;
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if !(-4..6).contains(&exp) {
        out.push_str(&d[..1]);
        if nd > 1 {
            out.push('.');
            out.push_str(&d[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let a = exp.abs();
        if a < 10 {
            out.push('0');
        }
        out.push_str(&a.to_string());
        return out;
    }
    // %f with max(nd-dp, 0) decimals.
    if dp > 0 {
        if nd >= dp {
            out.push_str(&d[..dp as usize]);
        } else {
            out.push_str(&d);
            out.push_str(&"0".repeat((dp - nd) as usize));
        }
    } else {
        out.push('0');
    }
    let decimals = (nd - dp).max(0);
    if decimals > 0 {
        out.push('.');
        for i in 0..decimals {
            let idx = dp + i;
            out.push(if idx < 0 {
                '0'
            } else {
                d.as_bytes()[idx as usize] as char
            });
        }
    }
    out
}

/// Go `%.4f` (correctly rounded, ties to even like Rust's formatter).
pub fn fmt_f4(x: f64) -> String {
    let s = format!("{x:.4}");
    // Go prints "-0.0000" for tiny negatives, as Rust does.
    s
}

// ---------------------------------------------------------------------------
// sort.Slice: Go's pdqsort_func, ported line for line (sort/zsortfunc.go).
// ---------------------------------------------------------------------------

/// Go `sort.Slice(v, less)`.
pub fn sort_slice<T>(v: &mut [T], less: impl Fn(&T, &T) -> bool) {
    let n = v.len();
    let limit = bits_len(n);
    let mut d = LessSwap { v, less: &less };
    pdqsort(&mut d, 0, n, limit);
}

/// Go `sort.SliceStable`: any stable sort gives the same order.
pub fn sort_slice_stable<T>(v: &mut [T], less: impl Fn(&T, &T) -> bool) {
    v.sort_by(|a, b| {
        if less(a, b) {
            Ordering::Less
        } else if less(b, a) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}

fn bits_len(n: usize) -> usize {
    (usize::BITS - n.leading_zeros()) as usize
}

struct LessSwap<'a, T, F: Fn(&T, &T) -> bool> {
    v: &'a mut [T],
    less: &'a F,
}

impl<T, F: Fn(&T, &T) -> bool> LessSwap<'_, T, F> {
    fn less(&self, i: usize, j: usize) -> bool {
        (self.less)(&self.v[i], &self.v[j])
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.v.swap(i, j);
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Hint {
    Unknown,
    Increasing,
    Decreasing,
}

fn insertion_sort<T, F: Fn(&T, &T) -> bool>(d: &mut LessSwap<T, F>, a: usize, b: usize) {
    for i in a + 1..b {
        let mut j = i;
        while j > a && d.less(j, j - 1) {
            d.swap(j, j - 1);
            j -= 1;
        }
    }
}

fn sift_down<T, F: Fn(&T, &T) -> bool>(d: &mut LessSwap<T, F>, lo: usize, hi: usize, first: usize) {
    let mut root = lo;
    loop {
        let mut child = 2 * root + 1;
        if child >= hi {
            break;
        }
        if child + 1 < hi && d.less(first + child, first + child + 1) {
            child += 1;
        }
        if !d.less(first + root, first + child) {
            return;
        }
        d.swap(first + root, first + child);
        root = child;
    }
}

fn heap_sort<T, F: Fn(&T, &T) -> bool>(d: &mut LessSwap<T, F>, a: usize, b: usize) {
    let first = a;
    let lo = 0;
    let hi = b - a;
    let mut i = (hi as isize - 1) / 2;
    while i >= 0 {
        sift_down(d, i as usize, hi, first);
        i -= 1;
    }
    let mut i = hi as isize - 1;
    while i >= 0 {
        d.swap(first, first + i as usize);
        sift_down(d, lo, i as usize, first);
        i -= 1;
    }
}

fn pdqsort<T, F: Fn(&T, &T) -> bool>(
    d: &mut LessSwap<T, F>,
    mut a: usize,
    mut b: usize,
    mut limit: usize,
) {
    const MAX_INSERTION: usize = 12;
    let mut was_balanced = true;
    let mut was_partitioned = true;
    loop {
        let length = b - a;
        if length <= MAX_INSERTION {
            insertion_sort(d, a, b);
            return;
        }
        if limit == 0 {
            heap_sort(d, a, b);
            return;
        }
        if !was_balanced {
            break_patterns(d, a, b);
            limit -= 1;
        }
        let (mut pivot, mut hint) = choose_pivot(d, a, b);
        if hint == Hint::Decreasing {
            reverse_range(d, a, b);
            pivot = (b - 1) - (pivot - a);
            hint = Hint::Increasing;
        }
        if was_balanced
            && was_partitioned
            && hint == Hint::Increasing
            && partial_insertion_sort(d, a, b)
        {
            return;
        }
        if a > 0 && !d.less(a - 1, pivot) {
            let mid = partition_equal(d, a, b, pivot);
            a = mid;
            continue;
        }
        let (mid, already) = partition(d, a, b, pivot);
        was_partitioned = already;
        let (left_len, right_len) = (mid - a, b - mid);
        let balance_threshold = length / 8;
        if left_len < right_len {
            was_balanced = left_len >= balance_threshold;
            pdqsort(d, a, mid, limit);
            a = mid + 1;
        } else {
            was_balanced = right_len >= balance_threshold;
            pdqsort(d, mid + 1, b, limit);
            b = mid;
        }
    }
}

fn partition<T, F: Fn(&T, &T) -> bool>(
    d: &mut LessSwap<T, F>,
    a: usize,
    b: usize,
    pivot: usize,
) -> (usize, bool) {
    d.swap(a, pivot);
    let (mut i, mut j) = (a as isize + 1, b as isize - 1);
    while i <= j && d.less(i as usize, a) {
        i += 1;
    }
    while i <= j && !d.less(j as usize, a) {
        j -= 1;
    }
    if i > j {
        d.swap(j as usize, a);
        return (j as usize, true);
    }
    d.swap(i as usize, j as usize);
    i += 1;
    j -= 1;
    loop {
        while i <= j && d.less(i as usize, a) {
            i += 1;
        }
        while i <= j && !d.less(j as usize, a) {
            j -= 1;
        }
        if i > j {
            break;
        }
        d.swap(i as usize, j as usize);
        i += 1;
        j -= 1;
    }
    d.swap(j as usize, a);
    (j as usize, false)
}

fn partition_equal<T, F: Fn(&T, &T) -> bool>(
    d: &mut LessSwap<T, F>,
    a: usize,
    b: usize,
    pivot: usize,
) -> usize {
    d.swap(a, pivot);
    let (mut i, mut j) = (a as isize + 1, b as isize - 1);
    loop {
        while i <= j && !d.less(a, i as usize) {
            i += 1;
        }
        while i <= j && d.less(a, j as usize) {
            j -= 1;
        }
        if i > j {
            break;
        }
        d.swap(i as usize, j as usize);
        i += 1;
        j -= 1;
    }
    i as usize
}

fn partial_insertion_sort<T, F: Fn(&T, &T) -> bool>(
    d: &mut LessSwap<T, F>,
    a: usize,
    b: usize,
) -> bool {
    const MAX_STEPS: usize = 5;
    const SHORTEST_SHIFTING: usize = 50;
    let mut i = a + 1;
    for _ in 0..MAX_STEPS {
        while i < b && !d.less(i, i - 1) {
            i += 1;
        }
        if i == b {
            return true;
        }
        if b - a < SHORTEST_SHIFTING {
            return false;
        }
        d.swap(i, i - 1);
        if i - a >= 2 {
            let mut j = i - 1;
            while j >= 1 {
                if !d.less(j, j - 1) {
                    break;
                }
                d.swap(j, j - 1);
                j -= 1;
            }
        }
        if b - i >= 2 {
            let mut j = i + 1;
            while j < b {
                if !d.less(j, j - 1) {
                    break;
                }
                d.swap(j, j - 1);
                j += 1;
            }
        }
    }
    false
}

fn break_patterns<T, F: Fn(&T, &T) -> bool>(d: &mut LessSwap<T, F>, a: usize, b: usize) {
    let length = b - a;
    if length >= 8 {
        let mut random = length as u64;
        let modulus = 1usize << bits_len(length);
        let mut idx = a + (length / 4) * 2 - 1;
        while idx <= a + (length / 4) * 2 + 1 {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            let mut other = (random as usize) & (modulus - 1);
            if other >= length {
                other -= length;
            }
            d.swap(idx, a + other);
            idx += 1;
        }
    }
}

fn choose_pivot<T, F: Fn(&T, &T) -> bool>(
    d: &mut LessSwap<T, F>,
    a: usize,
    b: usize,
) -> (usize, Hint) {
    const SHORTEST_NINTHER: usize = 50;
    const MAX_SWAPS: usize = 4 * 3;
    let l = b - a;
    let mut swaps = 0usize;
    let mut i = a + l / 4;
    let mut j = a + l / 4 * 2;
    let mut k = a + l / 4 * 3;
    if l >= 8 {
        if l >= SHORTEST_NINTHER {
            i = median_adjacent(d, i, &mut swaps);
            j = median_adjacent(d, j, &mut swaps);
            k = median_adjacent(d, k, &mut swaps);
        }
        j = median(d, i, j, k, &mut swaps);
    }
    match swaps {
        0 => (j, Hint::Increasing),
        MAX_SWAPS => (j, Hint::Decreasing),
        _ => (j, Hint::Unknown),
    }
}

fn order2<T, F: Fn(&T, &T) -> bool>(
    d: &LessSwap<T, F>,
    a: usize,
    b: usize,
    swaps: &mut usize,
) -> (usize, usize) {
    if d.less(b, a) {
        *swaps += 1;
        return (b, a);
    }
    (a, b)
}

fn median<T, F: Fn(&T, &T) -> bool>(
    d: &LessSwap<T, F>,
    a: usize,
    b: usize,
    c: usize,
    swaps: &mut usize,
) -> usize {
    let (a, b) = order2(d, a, b, swaps);
    let (b, c) = order2(d, b, c, swaps);
    let _ = c;
    let (_a, b) = order2(d, a, b, swaps);
    b
}

fn median_adjacent<T, F: Fn(&T, &T) -> bool>(
    d: &LessSwap<T, F>,
    a: usize,
    swaps: &mut usize,
) -> usize {
    median(d, a - 1, a, a + 1, swaps)
}

fn reverse_range<T, F: Fn(&T, &T) -> bool>(d: &mut LessSwap<T, F>, a: usize, b: usize) {
    let (mut i, mut j) = (a, b - 1);
    while i < j {
        d.swap(i, j);
        i += 1;
        j -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_v_formatting() {
        assert_eq!(fmt_v(1.0), "1");
        assert_eq!(fmt_v(0.0123), "0.0123");
        assert_eq!(fmt_v(1e-5), "1e-05");
        assert_eq!(fmt_v(0.00012345), "0.00012345");
        assert_eq!(fmt_v(123456.0), "123456");
        assert_eq!(fmt_v(1234567.0), "1.234567e+06");
        assert_eq!(fmt_v(1e6), "1e+06");
        assert_eq!(fmt_v(1791865240.977766), "1.791865240977766e+09");
        assert_eq!(fmt_v(-2.5), "-2.5");
        assert_eq!(fmt_v(0.1 + 0.2), "0.30000000000000004");
    }

    #[test]
    fn pdqsort_sorts() {
        let mut v: Vec<i32> = (0..200).map(|i| (i * 7919) % 101).collect();
        sort_slice(&mut v, |a, b| a < b);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn pairwise_matches_naive_below_16() {
        assert_eq!(pairwise_sum(&[0.1, 0.2, 0.3]), 0.1 + 0.2 + 0.3);
    }
}
