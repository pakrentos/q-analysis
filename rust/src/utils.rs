use std::cmp::Ordering;

pub fn compare_f64_nan_first(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).unwrap(),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}
