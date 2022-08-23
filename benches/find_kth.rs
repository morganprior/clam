pub mod utils;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use rand::Rng;
use std::cmp::Ordering;

use clam::prelude::*;
use clam::search::KnnSieve;
use clam::Tabular;

fn sort_and_index<'a>(v: &'a mut [(&clam::Cluster<'a, f64, f64>, f64)], k: usize) -> f64 {
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let m = v[k].1.clone();
    m
}

fn unstable_sort_and_index<'a>(v: &'a mut [(&clam::Cluster<'a, f64, f64>, f64)], k: usize) -> f64 {
    v.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    v[k].1.clone()
}

pub fn _find_kth(items: Vec<(&clam::Cluster<f64, f64>, f64)>, ccs: Vec<usize>, k: usize, l: usize, r: usize) -> f64 {
    let cards = (0..1)
        .chain(ccs.iter().cloned())
        .zip(ccs.iter().cloned())
        .map(|(prev, next)| next - prev)
        .collect();
    
    let (items, cards, partition_index) = partition(items, cards, l, r);

    match ccs[partition_index].cmp(&k) {
        Ordering::Less => _find_kth(items, ccs, k, partition_index + 1, r),
        Ordering::Equal => items[partition_index].1,
        Ordering::Greater => {
            if (partition_index > 0) && (ccs[partition_index -1] > k){
                _find_kth(items, ccs, k, l, partition_index - 1)
            } else {
                items[partition_index].1
            }
    }
} }

fn partition<'a>(
    items: Vec<(&'a clam::Cluster<'a, f64, f64>, f64)>,
    cards: Vec<usize>, 
    l: usize,
    r: usize,
) -> (Vec<(&'a clam::Cluster<'a, f64, f64>, f64)>, Vec<usize>, usize) {
    let mut items = items;
    let mut cards = cards;


    let pivot = (l + r) / 2; // Check for overflow
    swaps(&mut items, &mut cards, pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        if items[b].1 < items[r].1 {
            swaps(&mut items, &mut cards, a, b);
            a += 1;
        }
        b += 1;
    }

    swaps(&mut items, &mut cards, a, r);

    (items, cards, a)
}

fn swaps(i_1: &mut [(&clam::Cluster<f64, f64>, f64)], i_2: &mut [usize], a: usize, b: usize) {
    i_1.swap(a, b);
    i_2.swap(a, b); 
}

fn find_kth(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_kth");
    group.significance_level(0.05).sample_size(10);

    let mut rng = rand::thread_rng();
    let data: Vec<Vec<f64>> = (0..1000)
        .map(|_| (0..13).map(|_| rng.gen_range(0.0..100.0)).collect())
        .collect();
    let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
    let metric = metric_from_name::<f64, f64>("euclideansq", false).unwrap();
    let space = clam::TabularSpace::new(&dataset, metric.as_ref(), false);

    let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
    let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
    let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);
    let flat_tree = cluster.subtree();
    
    let sieve = KnnSieve::new(flat_tree.clone(), &data[0], 10).build();
    let items = sieve
        .clusters
        .iter()
        .zip(sieve.deltas.into_iter())
        .map(|(&c, d)| (c, d))
        .collect::<Vec<_>>();

    let diffs: Vec<usize> = (11..flat_tree.len() - 2).step_by(100).collect();

    for &diff in diffs.iter() {
        let mut clusters_with_duplicates: Vec<&Cluster<f64, f64>> = vec![];
        for distinct_cluster in &flat_tree[0..(diff + 1)] {
            for _ in 0..distinct_cluster.cardinality(){
                clusters_with_duplicates.push(distinct_cluster); }
        }
        let mut deltas_with_duplicates: Vec<f64> = vec![]; 
        for distinct_delta in sieve.deltas {
            for _ in 0..distinct_cluster.cardinality(){
                deltas_with_duplicates.push(distinct_delta); }
        }

        let kth_from_find_kth = _find_kth(items.clone()[0..(diff + 1)].to_vec(), sieve.cumulative_cardinalities.clone(), sieve.k, 0, diff);

        group.bench_with_input(BenchmarkId::new("find_kth", diff), &diff, |b, &diff| {
            b.iter(|| _find_kth(items.clone()[0..(diff + 1)].to_vec(), sieve.cumulative_cardinalities.clone(), sieve.k, 0, diff));
        });

        let kth_from_sort = sort_and_index(&mut items.clone()[0..(diff + 1)], sieve.k);

        group.bench_with_input(BenchmarkId::new("sort", diff), &diff, |b, &diff| {
            b.iter(|| sort_and_index(&mut clusters_with_duplicates.clone()[0..(diff + 1)], sieve.k));
        });

        group.bench_with_input(BenchmarkId::new("unstable_sort", diff), &diff, |b, &diff| {
            b.iter(|| unstable_sort_and_index(&mut clusters_with_duplicates.clone()[0..(diff + 1)], sieve.k));
        });
        assert_eq!(kth_from_find_kth, kth_from_sort);
    }
    group.finish();
}

criterion_group!(benches, find_kth);
criterion_main!(benches);
