use std::cmp::Ordering;

pub fn find_kth<'a, T: PartialOrd + Clone>(
    items: &'a [&'a T],
    cumulative_cardinalities: &'a [usize],
    k: usize,
) -> (Vec<&'a T>, Vec<usize>) {
    assert_eq!(items.len(), cumulative_cardinalities.len());
    _find_kth(items.to_vec(), cumulative_cardinalities.to_vec(), k, 0, items.len() - 1)
}

fn _find_kth<T: PartialOrd + Clone>(
    items: Vec<T>,
    cumulative_cardinalities: Vec<usize>,
    k: usize,
    l: usize,
    r: usize,
) -> (Vec<T>, Vec<usize>) {
    let cardinalities = (0..1)
        .chain(cumulative_cardinalities.iter().cloned())
        .zip(cumulative_cardinalities.iter().cloned())
        .map(|(prev, next)| next - prev)
        .collect();
    let (items, cardinalities, partition_index) = partition(items, cardinalities, l, r);

    let cumulative_cardinalities = cardinalities
        .iter()
        .scan(0, |acc, v| {
            *acc += *v;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    match cumulative_cardinalities[partition_index].cmp(&k) {
        Ordering::Less => _find_kth(items, cumulative_cardinalities, k, partition_index + 1, r),
        Ordering::Equal => (items, cumulative_cardinalities),
        Ordering::Greater => {
            if cumulative_cardinalities[partition_index - 1] > k {
                _find_kth(items, cumulative_cardinalities, k, l, partition_index - 1)
            } else {
                (items, cumulative_cardinalities)
            }
        }
    }
}

fn partition<T: PartialOrd + Clone>(
    items: Vec<T>,
    cardinalities: Vec<usize>,
    l: usize,
    r: usize,
) -> (Vec<T>, Vec<usize>, usize) {
    let mut items = items;
    let mut cardinalities = cardinalities;

    let pivot = (l + r) / 2; // Check for overflow
    swap_two(&mut items, &mut cardinalities, pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        if items[b] < items[r] {
            swap_two(&mut items, &mut cardinalities, a, b);
            a += 1;
        }
        b += 1;
    }

    todo!()
}

fn swap_two<T>(i: &mut [T], j: &mut [usize], a: usize, b: usize) {
    i.swap(a, b);
    j.swap(a, b);
}
