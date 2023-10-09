use criterion::*;

use symagen::random_data;

use abd_clam::{knn, Cakes, PartitionCriteria, VecDataset};

fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::vectors::euclidean(x, y)
}

const METRICS: &[(&str, fn(&Vec<f32>, &Vec<f32>) -> f32)] = &[("euclidean", euclidean)];

fn cakes(c: &mut Criterion) {
    let seed = 42;
    let (cardinality, dimensionality) = (1_000_000, 10);
    let (min_val, max_val) = (-1., 1.);

    let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);

    let num_queries = 100;
    let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);
    let queries = queries.iter().collect::<Vec<_>>();

    for &(metric_name, metric) in METRICS {
        let mut group = c.benchmark_group(format!("knn-{metric_name}"));
        group
            .sample_size(10)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(num_queries as u64))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        let dataset = VecDataset::new("knn".to_string(), data.clone(), metric, false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = Cakes::new(dataset, Some(seed), criteria);

        for k in (0..=8).map(|v| 2usize.pow(v)) {
            for &variant in knn::Algorithm::variants() {
                let id = BenchmarkId::new(variant.name(), k);
                group.bench_with_input(id, &k, |b, _| {
                    b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k, variant));
                });
            }
        }

        let id = BenchmarkId::new("Linear", 9);
        group.bench_with_input(id, &9, |b, _| {
            b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, 9, knn::Algorithm::Linear));
        });
        group.finish();
    }
}

criterion_group!(benches, cakes);
criterion_main!(benches);
