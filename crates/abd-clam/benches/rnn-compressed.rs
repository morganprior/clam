use criterion::*;

use rand::prelude::*;

use distances::strings::Penalties;

use symagen::random_edits::{are_we_there_yet, generate_clumped_data, generate_random_string};

use abd_clam::pancakes::{decode_general, encode_general, rnn, CodecData, SquishyBall};
use abd_clam::{Cakes, PartitionCriteria, VecDataset};

#[allow(clippy::ptr_arg)]
fn lev_metric(x: &String, y: &String) -> u16 {
    distances::strings::levenshtein(x, y)
}

fn compressed_rnn(c: &mut Criterion) {
    let seed = 42;
    let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect::<Vec<_>>();
    let seed_string = generate_random_string(100, &alphabet, seed);
    let penalties = Penalties::<u16>::new(0, 1, 1);
    let clump_radius = 15;
    let inter_clump_distance = 20;

    // (num_clumps, clump_size)
    let sizes = [
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (32, 64),
        (32, 128),
        (32, 256),
        (32, 512),
        (32, 1024),
    ];

    for (n, m) in sizes {
        let mut group = c.benchmark_group(format!("knn-{n}-{m}"));
        group
            // .sample_size(100)
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(1))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

        let clumped_data = generate_clumped_data(
            &seed_string,
            penalties,
            &alphabet,
            n,
            m,
            clump_radius,
            seed,
            inter_clump_distance,
        );

        let (queries, clumped_data) = {
            let mut clumped_data = clumped_data;
            // Shuffle the clumped data
            let mut rng: ThreadRng = rand::thread_rng();
            clumped_data.shuffle(&mut rng);
            let queries = clumped_data
                .iter()
                .take(1) // number of queries to take
                .map(|(m, p)| {
                    let q = are_we_there_yet(p, penalties, 5, &alphabet, &mut rand::rngs::StdRng::seed_from_u64(seed));
                    (m.clone(), q)
                })
                .collect::<Vec<_>>();
            (queries, clumped_data)
        };

        let (_, query_data): (Vec<_>, Vec<_>) = queries.into_iter().unzip();
        let (_, clumped_data): (Vec<_>, Vec<_>) = clumped_data.into_iter().unzip();

        let name = format!("{n}x{m}");
        let dataset = VecDataset::new(name, clumped_data, lev_metric, true);
        let criteria = PartitionCriteria::default();

        let cakes = Cakes::new(dataset, Some(seed), &criteria);

        let cakes_tree = cakes.trees()[0];
        let dataset = cakes_tree.data();
        let root = cakes_tree.root().clone();
        let root = SquishyBall::from_base_tree(root, dataset);

        let metadata = dataset.metadata().to_vec();
        let compressed_dataset = CodecData::new(root, dataset, encode_general::<u16>, decode_general, metadata).unwrap();

        for query in query_data {
            for radius in [4, 8, 16] {
                let id = BenchmarkId::new("Clustered", radius);
                group.bench_with_input(id, &radius, |b, _| {
                    b.iter_with_large_drop(|| compressed_dataset.rnn_search(&query, radius, &rnn::Algorithm::Clustered));
                });

                let id = BenchmarkId::new("Linear", radius);
                group.bench_with_input(id, &radius, |b, _| {
                    b.iter_with_large_drop(|| compressed_dataset.rnn_search(&query, radius, &rnn::Algorithm::Linear));
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, compressed_rnn);
criterion_main!(benches);
