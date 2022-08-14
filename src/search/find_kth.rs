use std::cmp::Ordering;

pub fn find_kth<'a, T: PartialOrd + Clone, U: PartialOrd + Clone>(
    items: &'a [T],
    cumulative_cardinalities: &'a [usize],
    deltas: &'a [U],
    k: usize,
) -> (Vec<T>, Vec<U>, Vec<usize>) {
    assert_eq!(items.len(), cumulative_cardinalities.len());
    _find_kth(
        items.to_vec(),
        cumulative_cardinalities.to_vec(),
        deltas.to_vec(),
        k,
        0,
        items.len() - 1,
    )
}

fn _find_kth<T: PartialOrd + Clone, U: PartialOrd + Clone>(
    items: Vec<T>,
    cumulative_cardinalities: Vec<usize>,
    deltas: Vec<U>,
    k: usize,
    l: usize,
    r: usize,
) -> (Vec<T>, Vec<U>, Vec<usize>) {
    let cardinalities = (0..1)
        .chain(cumulative_cardinalities.iter().cloned())
        .zip(cumulative_cardinalities.iter().cloned())
        .map(|(prev, next)| next - prev)
        .collect();
    let (items, cardinalities, deltas, partition_index) = partition(items, cardinalities, deltas, l, r);

    let cumulative_cardinalities = cardinalities
        .iter()
        .scan(0, |acc, v| {
            *acc += *v;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    match cumulative_cardinalities[partition_index].cmp(&k) {
        Ordering::Less => _find_kth(items, cumulative_cardinalities, deltas, k, partition_index + 1, r),
        Ordering::Equal => (items, deltas, cumulative_cardinalities),
        Ordering::Greater => {
            if (partition_index > 1) && (cumulative_cardinalities[partition_index - 1] > k) {
                _find_kth(items, cumulative_cardinalities, deltas, k, l, partition_index - 1)
            } else {
                (items, deltas, cumulative_cardinalities)
            }
        }
    }
}

fn partition<T: PartialOrd + Clone, U: PartialOrd + Clone>(
    items: Vec<T>,
    cardinalities: Vec<usize>,
    deltas: Vec<U>,
    l: usize,
    r: usize,
) -> (Vec<T>, Vec<usize>, Vec<U>, usize) {
    let mut items = items;
    let mut cardinalities = cardinalities;
    let mut deltas = deltas;

    let pivot = (l + r) / 2; // Check for overflow
    swaps(&mut items, &mut cardinalities, &mut deltas, pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        if items[b] < items[r] {
            swaps(&mut items, &mut cardinalities, &mut deltas, a, b);
            a += 1;
        }
        b += 1;
    }

    (items, cardinalities, deltas, a)
}

fn swaps<T, U>(i_1: &mut [T], i_2: &mut [usize], i_3: &mut [U], a: usize, b: usize) {
    i_1.swap(a, b);
    i_2.swap(a, b);
    i_3.swap(a, b);
}
