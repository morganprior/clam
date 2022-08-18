pub mod utils;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use rand::Rng;

use clam::prelude::*;
use clam::search::find_kth;
use clam::search::KnnSieve;
use clam::Tabular;

fn find_kth(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_kth");
    group
        .significance_level(0.05)
        .sample_size(10);

    
    let mut rng = rand::thread_rng();
    let data: Vec<Vec<f64>> = (0..1000).map(|_| (0..3).map(|_| rng.gen_range(0.0..20.0)).collect()).collect();
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
    let other_deltas = sieve
        .deltas_max
        .into_iter()
        .zip(sieve.deltas_min.into_iter())
        .collect::<Vec<_>>();

    let diffs: Vec<usize> = ((sieve.k)..flat_tree.len() - 2).step_by(30).collect();

    for &diff in diffs.iter() {
        group.bench_with_input(BenchmarkId::new("find_kth", diff), &diff, |b, &diff| {
            b.iter(|| {
                find_kth::_find_kth(
                    items.clone(),
                    sieve.cumulative_cardinalities.clone(),
                    other_deltas.clone(),
                    sieve.k,
                    0,
                    diff,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, find_kth);
criterion_main!(benches);
