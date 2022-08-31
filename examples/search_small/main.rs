use std::collections::HashSet;

// use rayon::prelude::*;

use clam::prelude::*;
use clam::utils::helpers;

mod h5data;
mod h5number;
mod h5space;

pub fn open_hdf5_file(name: &str) -> hdf5::Result<hdf5::File> {
    let mut data_dir = std::env::current_dir().unwrap();
    data_dir.pop();
    data_dir.push("data");
    data_dir.push("search_small");
    data_dir.push("as_hdf5");

    let train_path = {
        let mut train_path = data_dir.clone();
        train_path.push(format!("{}.hdf5", name));
        assert!(train_path.exists(), "{:?} does not exist.", &train_path);
        train_path
    };

    hdf5::File::open(train_path)
}

// An example for a custom partition criterion.
#[derive(Debug, Clone)]
struct MinRadius<U: Number> {
    threshold: U,
}

impl<T: Number, U: Number> clam::PartitionCriterion<T, U> for MinRadius<U> {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.radius() > self.threshold
    }
}

fn compute_recall(hits: &[Vec<usize>], true_hits: &[HashSet<usize>]) -> Vec<f64> {
    let hits: Vec<HashSet<usize>> = hits.iter().map(|r| HashSet::from_iter(r.iter().copied())).collect();
    let recalls: Vec<f64> = hits
        .iter()
        .zip(true_hits)
        .map(|(pred, actual)| {
            let intersection = pred.intersection(actual).count();
            (intersection as f64) / (actual.len() as f64)
        })
        .collect();
    recalls
}

#[allow(clippy::single_element_loop)]
fn search<Tr, Te, N, D, T>(data_name: &str, metric_name: &str, num_runs: usize) -> Result<(), String>
where
    Tr: h5number::H5Number, // For reading "train" data from hdf5 files.
    Te: h5number::H5Number, // For reading "test" data from hdf5 files.
    N: h5number::H5Number,  // For reading ground-truth neighbors' indices.
    D: h5number::H5Number,  // For reading ground-truth neighbors' distances.
    T: Number,              // For converting Tr and Te away from bool for kosarak data.
{
    log::info!("");
    log::info!("Running {}-{} data ...", data_name, metric_name);

    let file =
        open_hdf5_file(data_name).map_err(|reason| format!("Could not read file {} because {}", data_name, reason))?;

    let neighbors =
        h5data::H5Data::<N>::new(&file, "neighbors", format!("{}_neighbors", data_name))?.to_vec_vec::<usize>()?;

    let distances =
        h5data::H5Data::<D>::new(&file, "distances", format!("{}_distances", data_name))?.to_vec_vec::<D>()?;

    let search_radii: Vec<_> = distances
        .into_iter()
        .map(|row| clam::utils::helpers::arg_max(&row).1)
        .collect();

    let queries = h5data::H5Data::<Te>::new(&file, "test", format!("{}_test", data_name))?.to_vec_vec::<T>()?;
    let queries = clam::Tabular::new(&queries, format!("{}-queries", data_name));
    // let num_queries = queries.cardinality();
    let num_queries = 10;
    let queries = (0..num_queries)
        .map(|i| queries.get(i % queries.cardinality()))
        .collect::<Vec<_>>();
    // let queries = vec![queries.get(42)];
    // let num_queries = queries.len();

    let metric = clam::metric_from_name::<T, D>(metric_name, false)?;

    let train = h5data::H5Data::<Tr>::new(&file, "train", "temp".to_string())?;

    // because reading h5 data with this crate is too slow ...
    // let space = h5space::H5Space::new(&train, metric.as_ref(), false);
    let train = train.to_vec_vec::<T>()?;
    let train = clam::Tabular::new(&train, data_name.to_string());
    let space = clam::TabularSpace::new(&train, metric.as_ref(), false);

    let partition_criteria = clam::PartitionCriteria::default();

    log::info!("Building search tree on {}-{} data ...", data_name, metric_name);

    let start = std::time::Instant::now();
    let cakes = clam::CAKES::new(&space).build(&partition_criteria);
    let build_time = start.elapsed().as_secs_f64();
    log::info!(
        "Built tree to a depth of {} in {:.2e} seconds ...",
        cakes.depth(),
        build_time
    );

    let min_r = helpers::arg_min(&search_radii).1.as_f64() / cakes.radius().as_f64();
    let max_r = helpers::arg_max(&search_radii).1.as_f64() / cakes.radius().as_f64();
    let mean_r = helpers::mean(&search_radii) / cakes.radius().as_f64();
    let sd_r = helpers::sd(&search_radii, mean_r) / cakes.radius().as_f64();
    log::info!(
        "True search-radii fractions' range is [{:.2e}, {:.2e}] with mean {:.2e} and sd {:.2e}",
        min_r,
        max_r,
        mean_r,
        sd_r
    );

    log::info!("");
    log::info!(
        "Starting knn-search on {}-{} data with {} queries ...",
        data_name,
        metric_name,
        num_queries
    );
    log::info!("");

    // fashion-mnist search times (ms per query) for 1_000 queries:
    // (double-pq):       29.4, 62.2, 65.1   //
    // (single-pq):       30.0, 55.5, 58.0   //
    // (small-insiders):  30.2, 41.8, 57.9   //

    // (sorting) multi-threaded search times (ms per query) for 1_000 queries
    // deep-image     , 105.          , 307.          , 387.
    // fashion-mnist  ,   4.58        ,   6.78        ,   8.77
    // gist           , 285.          , 297.          , 305.
    // glove-25       ,   8.81        ,  17.3         ,  22.7
    // glove-50       ,  51.4         ,  74.6         ,  96.4
    // glove-100      , 132.          , 156.          , 165.
    // glove-200      , 200.          , 212.          , 217.
    // lastfm         ,    .000161    ,    .000980    ,  31.9
    // mnist          ,   8.51        ,  10.9         ,  13.1
    // nytimes        ,  34.6         ,  41.3         ,  41.2
    // sift           ,  45.1         ,  57.5         ,  67.5

    // (find_kth) multi-threaded search times (ms per query) for 1_000 queries
    // deep-image     ,  85.3         , 300.          , 355.          //
    // fashion-mnist  ,   3.56        ,   4.78        ,   6.70        //
    // gist           , 257.          , 267.          , 274.          //
    // glove-25       ,   3.89        ,  10.7         ,  14.7         //
    // glove-50       ,  37.7         ,  53.6         ,  72.2         //
    // glove-100      , 121.          , 129.          , 139.          //
    // glove-200      , 184.          , 192.          , 193.          //
    // lastfm         ,    .000655    ,    .00153     ,  34.7         //
    // mnist          ,   8.21        ,  10.2         ,  12.0         //
    // nytimes        ,    .          ,    .          ,    .          // Stack-overflow error from recursion in find_kth. Tree was 254 deep.
    // sift           ,  37.3         ,  43.2         ,  52.0         //

    // let ks = [1, 10, 100];
    let ks = [100];
    for k in ks {
        log::info!("Using k = {} ...", k);
        log::info!("");

        let start = std::time::Instant::now();
        // let knn_hits = (0..num_queries)
        //     .map(|_| {
        //         queries
        //             .par_iter()
        //             .zip(search_radii.par_iter())
        //             .enumerate()
        //             .map(|(i, (&query, &radius))| (i, cakes.rnn_search(query, radius)))
        //             .inspect(|(i, _)| log::debug!("Finished query {}/{} ...", i, num_queries))
        //             .map(|(_, hits)| hits)
        //             .collect::<Vec<_>>()
        //     })
        //     .last()
        //     .unwrap();
        let knn_hits = (0..num_runs)
            .map(|_| cakes.batch_knn_search(&queries, k))
            .last()
            .unwrap();
        let time = start.elapsed().as_secs_f64() / (num_runs as f64);
        let mean_time = time / (num_queries as f64);

        // let knn_hits: Vec<Vec<usize>> = knn_hits
        //     .into_iter()
        //     .map(|hits| hits.into_iter().map(|(i, _)| i).collect())
        //     .collect();

        log::info!("knn-search time: {:.2e} seconds per query ...", mean_time);
        log::info!("");

        if k == neighbors[0].len() {
            let true_hits = neighbors
                .iter()
                .map(|row| row.iter().copied().collect())
                .collect::<Vec<_>>();

            let recalls = compute_recall(&knn_hits, &true_hits);

            let mean_recall = helpers::mean(&recalls);
            let sd_recall = helpers::sd(&recalls, mean_recall);
            log::info!("knn-recall: {:.2e} +/- {:.2e} ...", mean_recall, sd_recall);
            log::info!("");
        }
    }
    log::info!("Moving on ...");
    log::info!("");

    Ok(())
}

fn main() -> Result<(), String> {
    env_logger::Builder::new().parse_filters("info").init();

    let results = [
        search::<f32, f32, i32, f32, f32>("deep-image", "cosine", 1),
        // search::<f32, f32, i32, f32, f32>("fashion-mnist", "euclidean", 1),
        // search::<f32, f32, i32, f32, f32>("gist", "euclidean", 1),
        // search::<f32, f32, i32, f32, f32>("glove-25", "cosine", 1),
        // search::<f32, f32, i32, f32, f32>("glove-50", "cosine", 1),
        // search::<f32, f32, i32, f32, f32>("glove-100", "cosine", 1),
        // search::<f32, f32, i32, f32, f32>("glove-200", "cosine", 1),
        // search::<f32, f64, i32, f32, f32>("lastfm", "cosine", 1),
        // search::<f32, f32, i32, f32, f32>("mnist", "euclidean", 1),
        // search::<f32, f32, i32, f32, f32>("nytimes", "cosine", 1),
        // search::<f32, f32, i32, f32, f32>("sift", "euclidean", 1),
        // search::<bool, bool, i32, f32, u8>("kosarak", "jaccard", 1),
    ];
    println!(
        "Successful for {}/{} datasets.",
        results.iter().filter(|v| v.is_ok()).count(),
        results.len()
    );
    let failures: Vec<_> = results.iter().filter(|v| v.is_err()).cloned().collect();
    if !failures.is_empty() {
        println!("Failed for {}/{} datasets.", failures.len(), results.len());
        failures.into_iter().for_each(|v| println!("{:?}\n", v));
    }
    Ok(())
}
