use std::cmp::Ordering;

use super::knn_sieve;

use crate::prelude::*;

/// Public function to find the kth Grain if `grains` were sorted by Delta::D (distance from Grain to query)
/// 
/// This function just calls _find_kth, which does all the grunt work 
/// 
/// Returns the kth Grain and its index in `grains`.
pub fn find_kth_d<'a, T: Number, U: Number>(
    grains: &mut [knn_sieve::Grain<'a, T, U>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
) -> (knn_sieve::Grain<'a, T, U>, usize) {
    assert_eq!(grains.len(), cumulative_cardinalities.len());
    _find_kth(
        grains,
        cumulative_cardinalities,
        k,
        0,
        grains.len() - 1,
        &knn_sieve::Delta::D,
    )
}

/// Public function to find the kth Grain if `grains` were sorted by Delta::Max (distance from 
/// query to the potentially furthest point in Grain)
/// 
/// This function just calls _find_kth, which does all the grunt work 
/// 
/// Returns the kth Grain and its index in `grains`.
pub fn find_kth_d_max<'a, T: Number, U: Number>(
    grains: &mut [knn_sieve::Grain<'a, T, U>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
) -> (knn_sieve::Grain<'a, T, U>, usize) {
    assert_eq!(grains.len(), cumulative_cardinalities.len());
    _find_kth(
        grains,
        cumulative_cardinalities,
        k,
        0,
        grains.len() - 1,
        &knn_sieve::Delta::Max,
    )
}

/// Public function to find the kth Grain if `grains` were sorted by Delta::Min (distance from 
/// query to the potentially closest point in Grain)
/// 
/// This function just calls _find_kth, which does all the grunt work 
/// 
/// Returns the kth Grain and its index in `grains`.
pub fn find_kth_d_min<'a, T: Number, U: Number>(
    grains: &mut [knn_sieve::Grain<'a, T, U>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
) -> (knn_sieve::Grain<'a, T, U>, usize) {
    assert_eq!(grains.len(), cumulative_cardinalities.len());
    _find_kth(
        grains,
        cumulative_cardinalities,
        k,
        0,
        grains.len() - 1,
        &knn_sieve::Delta::Min,
    )
}


/// Finds the kth `Grain` if `grains` were sorted by `delta` and if each `Grain` occured with 
/// multiplicity equal to the cardinality of its `cluster` member. (I.e., if a Grain's `cluster` 
/// member is a Cluster with cardinality 7, _find_kth operates as if that Grain appears 7 consecutive times 
/// in `grains`.)
/// (A `grain` is a struct whose members are a Cluster and the `delta`, 
/// `delta_min`, and `delta_max` of that Cluster.) 
///  
/// Returns tuple of the kth grain and its index in `grains`.
///
/// # Arguments
///
/// * `grains`: Vector of `grain`s, sorted by distance to query
/// * `cumulative_cardinalities`: Vector of cardinalities corresponding to `grains`. The ith element 
/// of `cumulative_cardinalities` represents the sum of cardinalities of the 0th through ith `grain` in
/// `grains`
/// * `k`: index of desired element if list of grains were sorted by `delta`
/// * `l`: index of leftmost element (0 for first call, can increase for subsequent recursive calls)
/// * `r`: index of rightmost element (length of grains for first call, can decrease for subsequent recursive calls
/// * `delta`: type of delta on which to base ordering of grains (can be Delta::Max, Delta::Min, or Delta::D). 
/// Delta::Max is the distance from the query to the furthest point in the cluster, Delta::Min is the distance from 
/// the query to the closest point in the cluster, and Delta::D is the distance from the query to the cluster center
pub fn _find_kth<'a, T: Number, U: Number>(
    grains: &mut [knn_sieve::Grain<'a, T, U>],
    cumulative_cardinalities: &mut [usize],
    k: usize,
    l: usize,
    r: usize,
    delta: &knn_sieve::Delta,
) -> (knn_sieve::Grain<'a, T, U>, usize) {

    // We partition the list of grains based on individual cluster cardinalities, NOT 
    // cumulative cardinalities, so the first step here is to map cumulative cardinalities to 
    // individual cluster cardinalities by taking the difference between consecutive cumuluative 
    // cardinalities. cardinalities[i] is the cardinality of the cluster associated with the ith grain. 
    let mut cardinalities = (0..1)
        .chain(cumulative_cardinalities.iter().copied())
        .zip(cumulative_cardinalities.iter().copied())
        .map(|(prev, next)| next - prev)
        .collect::<Vec<_>>();
    assert_eq!(cardinalities.len(), grains.len());

    // position is the "true" index (i.e., index if `grains` were sorted by `delta`) of the Grain currently 
    // at the rightmost possible index that could still have the `k`th Grain (i.e.,`r`th index). 
    let position = partition(grains, &mut cardinalities, l, r, delta);

    // Since partition() modifies its arguments, we need to recompute cumulative_cardinalities 
    // to reflect swaps that may have happened in cardinalities
    let mut cumulative_cardinalities = cardinalities
        .iter()
        .scan(0, |acc, v| {
            *acc += *v;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    // Since this function operates as if each Grain appears in `grains` with multiplicity equal to its 
    // cardinality, we compare k to the cumulative cardinality up to and including the Grain with 
    // index = position in grains. 
    // 
    // If the cumulative cardinality is less, we know the kth Grain occurs 
    // at an index greater than position, so this function calls itself with adjusted left and right indices. 
    //
    // If the cumulative cardinality exactly equals k, then the kth Grain is the Grain at index = position 
    //
    // If the cumulative cardinality is greater than k, since each Grain appears with multiplicity according to 
    // its cardinality, the Grain at index = position may be the desired Grain, or it may not, depending on how much 
    // greater the cumulative cardinality is. If the Grain at index = position - 1 has cumulative cardinality less than 
    // or equal to k, the Grain at index = position is the kth Grain. If not, the kth Grain occurs at an index lower 
    // than position, so this function calls itself with adjusted left and right indices. 
    match cumulative_cardinalities[position].cmp(&k) {
        Ordering::Less => _find_kth(grains, &mut cumulative_cardinalities, k, position + 1, r, delta),
        Ordering::Equal => (grains[position].clone(), position),
        Ordering::Greater => {
            if (position > 0) && (cumulative_cardinalities[position - 1] > k) {
                _find_kth(grains, &mut cumulative_cardinalities, k, l, position - 1, delta)
            } else {
                (grains[position].clone(), position)
            }
        }
    }
}

/// Returns the "true" index (i.e., index if `grains` were sorted by `delta`) of the Grain currently 
/// at the rightmost possible index that could still have the `k`th Grain (i.e.,`r`th index). 
///
/// `j` tracks the index of the Grain being evaluated at each iteration of the while loop. `i`
/// counts the number of Grains whose delta value is less than that of the `r`th Grain.
///
/// If a Grain has a delta value greater than or equal to that of the `r`th Grain, it  
/// swaps position with the next Grain whose delta is less than that of the `r`th Grain.
///
/// Since `i` counts the number of Grain with a delta less than the `r`th Cluster, the final
/// swap of the `i`th and `r`th Grains puts that `r`th Grain in its correct position (as if
/// `grains` were sorted by `delta`).
/// 
/// NOTE: This modifies (via swaps) both `grains` and `cardinalities`
fn partition<'a, T: Number, U: Number>(
    grains: &mut [knn_sieve::Grain<'a, T, U>],
    cardinalities: &mut [usize],
    l: usize,
    r: usize,
    delta: &knn_sieve::Delta,
) -> usize {

    // Selects pivot point to be the approximate middle element and swaps element at that location 
    // with element at the rightmost location. 
    let pivot = (l + r) / 2; 
    swaps(grains, cardinalities, pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        let grain_comp = match delta {
            knn_sieve::Delta::D => (grains[b]).ord_by_d(&grains[r]),
            knn_sieve::Delta::Max => (grains[b]).ord_by_d_max(&grains[r]),
            knn_sieve::Delta::Min => (grains[b]).ord_by_d_min(&grains[r]),
        };

        if grain_comp == Ordering::Less {
            swaps(grains, cardinalities, a, b);
            a += 1;
        }
        b += 1;
    }

    swaps(grains, cardinalities, a, r);

    a
}

pub fn swaps<T>(grains: &mut [T], cardinalities: &mut [usize], i: usize, j: usize) {
    grains.swap(i, j);
    cardinalities.swap(i, j);
}

/// This function does essentially the same thing as the find_kth functions above, but with slight modifications
/// because thresholds are tuples of distance values and cardinalities. 
pub fn find_kth_threshold<U: Number>(thresholds: &mut [(U, usize)], k: usize) -> (usize, (U, usize)) {
    let (index, (threshold, _)) = _find_kth_threshold(thresholds, k, 0, thresholds.len() - 1);

    // The kth threshold identified in the previous line may not be unique; other thresholds may be the 
    // same distance away from the query. We want to capture all of those thresholds; hence, we want to 
    // increase our threshold index to include of them, AND we want them to be in the correct position in 
    // thresholds
    let mut b = index;
    for a in (index + 1)..(thresholds.len()) {
        if thresholds[a].0 == threshold {
            b += 1;
            thresholds.swap(a, b);
        }
    }

    (b, thresholds[b])
}

fn _find_kth_threshold<U: Number>(thresholds: &mut [(U, usize)], k: usize, l: usize, r: usize) -> (usize, (U, usize)) {
    if l >= r {
        let position = std::cmp::min(l, r);
        (position, thresholds[position])
    } else {
        let position = partition_threshold(thresholds, l, r);
        let guaranteed_cardinalities = thresholds
            .iter()
            .scan(0, |acc, &(_, v)| {
                *acc += v;
                Some(*acc)
            })
            .collect::<Vec<_>>();
        assert!(
            guaranteed_cardinalities[r] > k,
            "Too few guarantees {} vs {} at {} ...",
            guaranteed_cardinalities[r],
            k,
            r
        );

        let num_guaranteed = guaranteed_cardinalities[position];

        match num_guaranteed.cmp(&k) {
            Ordering::Less => _find_kth_threshold(thresholds, k, position + 1, r),
            Ordering::Equal => (position, thresholds[position]),
            Ordering::Greater => {
                if (position > 0) && (guaranteed_cardinalities[position - 1] > k) {
                    _find_kth_threshold(thresholds, k, l, position - 1)
                } else {
                    (position, thresholds[position])
                }
            }
        }
    }
}

pub fn partition_threshold<U: Number>(thresholds: &mut [(U, usize)], l: usize, r: usize) -> usize {
    let pivot = (l + r) / 2; // Check for overflow
    thresholds.swap(pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        if thresholds[b].0 <= thresholds[r].0 {
            thresholds.swap(a, b);
            a += 1;
        }
        b += 1;
    }

    thresholds.swap(a, r);

    a
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn find_kth_vs_sort() {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..1000)
            .map(|_| (0..5).map(|_| rng.gen_range(0.0..10.0)).collect())
            .collect();
        let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
        let metric = metric_from_name::<f64, f64>("euclideansq", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref(), false);

        let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = crate::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);
        let flat_tree = cluster.subtree();

        let query_ind = rng.gen_range(0..data.len());
        let k = 42;

        let mut sieve = knn_sieve::KnnSieve::new(flat_tree.clone(), &data[query_ind], k);

        let kth_grain = find_kth_d_max(&mut sieve.grains, &mut sieve.cumulative_cardinalities, sieve.k);

        sieve.grains.sort_by(|a, b| a.ord_by_d_max(b));
        sieve.update_cumulative_cardinalities();
        sieve.update_guaranteed_cardinalities();
        let index = sieve
            .cumulative_cardinalities
            .iter()
            .position(|c| *c >= sieve.k)
            .unwrap();

        let threshold_from_sort = sieve.grains[index].d_max;
        assert_eq!(kth_grain.0.d_max, threshold_from_sort);
    }

    #[test]
    fn kth_relative_position() {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..1000)
            .map(|_| (0..5).map(|_| rng.gen_range(0.0..10.0)).collect())
            .collect();
        let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
        let metric = metric_from_name::<f64, f64>("euclideansq", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref(), false);

        let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = crate::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);
        let flat_tree = cluster.subtree();

        let query_ind = rng.gen_range(0..data.len());
        let k = 42;

        let mut sieve = knn_sieve::KnnSieve::new(flat_tree.clone(), &data[query_ind], k);

        let kth_grain = find_kth_d_min(&mut sieve.grains, &mut sieve.cumulative_cardinalities, sieve.k);

        for i in 0..sieve.grains.clone().len() {
            if i < kth_grain.1 {
                assert!(sieve.grains[i].d_min <= kth_grain.0.d_min);
            } else {
                assert!(sieve.grains[i].d_min >= kth_grain.0.d_min);
            }
        }
    }
}
