pub mod utils;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use std::cmp::Ordering;
use rand::Rng;

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



pub fn _find_kth(
    items: Vec<(&clam::Cluster<f64, f64>, f64)>,
    k: usize,
    l: usize,
    r: usize,
) -> f64 {

    let (items, partition_index) = partition(items,  l, r);

   
    match partition_index.cmp(&k) {
        Ordering::Less => _find_kth(items, k, partition_index + 1, r),
        Ordering::Equal => items[k].1, 
        Ordering::Greater => _find_kth(items, k, l, partition_index - 1), 
            } 
        }


fn partition<'a>(
    items: Vec<(&'a clam::Cluster<'a, f64, f64>, f64)>,
    l: usize,
    r: usize,
) -> (Vec<(&'a clam::Cluster<'a, f64, f64>, f64)>,  usize) {
    let mut items = items;

    let pivot = (l + r) / 2; // Check for overflow
    swaps(&mut items,  pivot, r);

    let (mut a, mut b) = (l, l);
    while b < r {
        if items[b].1 < items[r].1 {
            swaps(&mut items,  a, b);
            a += 1;
        }
        b += 1;
    }

    swaps(&mut items, a, r);

    (items, a)
}

fn swaps(i_1: &mut [(&clam::Cluster<f64, f64>, f64)],  a: usize, b: usize) {
    i_1.swap(a, b);
}

fn find_kth(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_kth");
    group.significance_level(0.05).sample_size(10);

    let mut rng = rand::thread_rng();
    let data: Vec<Vec<f64>> = (0..1000000)
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
  
    let diffs: Vec<usize> = (11..flat_tree.len() - 2).step_by(5000).collect();
    
    for &diff in diffs.iter() {
        group.bench_with_input(BenchmarkId::new("find_kth", diff), &diff, |b, &diff| {
            b.iter(|| {
                _find_kth(
                    items.clone()[0..(diff+1)].to_vec(),
                    sieve.k, 
                    0, 
                    diff

                )
            });
        });
        let kth_from_find_kth = _find_kth(
            items.clone()[0..(diff+1)].to_vec(),
            sieve.k, 
            0, 
            diff); 

        
        group.bench_with_input(BenchmarkId::new("sort", diff), &diff, |b, &diff| {
            b.iter(|| sort_and_index(&mut items.clone()[0..(diff+1)], sieve.k));
        });

        let kth_from_sort = sort_and_index(&mut items.clone()[0..(diff+1)], sieve.k);

        group.bench_with_input(BenchmarkId::new("unstable_sort", diff), &diff, |b, &diff| {
            b.iter(|| unstable_sort_and_index(&mut items.clone()[0..(diff+1)], sieve.k));
        });
        assert_eq!(kth_from_find_kth, kth_from_sort); 
    }
    group.finish();
}

criterion_group!(benches, find_kth);
criterion_main!(benches);
