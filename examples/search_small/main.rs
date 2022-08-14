use std::collections::HashSet;

use rayon::prelude::*;

use clam::prelude::*;

mod h5data;
mod h5number;
mod h5space;
mod utils;

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

fn search<Tr, Te, N, D, T>(data_name: &str, metric_name: &str, num_runs: usize) -> Result<(), String>
where
    Tr: h5number::H5Number, // For reading "train" data from hdf5 files.
    Te: h5number::H5Number, // For reading "test" data from hdf5 files.
    N: h5number::H5Number,  // For reading ground-truth neighbors' indices.
    D: h5number::H5Number,  // For reading ground-truth neighbors' distances.
    T: Number,              // For converting Tr and Te away from bool for kosarak data.
{
    let output_dir = {
        let mut output_dir = std::env::current_dir().unwrap();
        output_dir.pop();
        output_dir.push("results");
        assert!(output_dir.exists(), "Path not found: {:?}", output_dir);

        output_dir.push(data_name);
        utils::make_dir(&output_dir, false)?;

        output_dir.push(metric_name);
        utils::make_dir(&output_dir, false)?;

        output_dir
    };

    log::info!("Running {}-{} data ...", data_name, metric_name);

    let file =
        open_hdf5_file(data_name).map_err(|reason| format!("Could not read file {} because {}", data_name, reason))?;

    let neighbors =
        h5data::H5Data::<N>::new(&file, "neighbors", format!("{}_neighbors", data_name))?.to_vec_vec::<usize>()?;

    let distances =
        h5data::H5Data::<D>::new(&file, "distances", format!("{}_distances", data_name))?.to_vec_vec::<D>()?;

    // let search_radii: Vec<_> = distances
    //     .into_iter()
    //     .map(|row| clam::utils::helpers::arg_max(&row).1)
    //     .collect();

    // Adding a 10% buffer to search radius to test change in recall
    let search_radii: Vec<_> = distances
        .into_iter()
        .map(|row| clam::utils::helpers::arg_max(&row).1)
        .map(|v| D::from(v.as_f64() * 1.1).unwrap())
        .collect();

    let min_radius = clam::utils::helpers::arg_min(&search_radii).1;

    let queries = h5data::H5Data::<Te>::new(&file, "test", format!("{}_test", data_name))?.to_vec_vec::<T>()?;
    // let queries = queries[0..100].to_vec();

    let queries_radii: Vec<(Vec<T>, D)> = queries.into_iter().zip(search_radii.iter().cloned()).collect();

    let metric = clam::metric_from_name::<T, D>(metric_name, false)?;

    let train = h5data::H5Data::<Tr>::new(&file, "train", "temp".to_string())?;

    // because reading h5 data with this crate is too slow ...
    // let space = h5space::H5Space::new(&train, metric.as_ref(), false);
    let train = train.to_vec_vec::<T>()?;
    let train = clam::Tabular::new(&train, data_name.to_string());
    let space = clam::TabularSpace::new(&train, metric.as_ref(), false);

    // let log_cardinality = (space.data().cardinality() as f64).log2() as usize;
    let partition_criteria = clam::PartitionCriteria::new(true)
        .with_min_cardinality(10)
        .with_custom(Box::new(MinRadius {
            threshold: D::from(min_radius.as_f64() / 1000.).unwrap(),
        }));

    log::info!("Building search tree on {}-{} data ...", data_name, metric_name);

    let start = std::time::Instant::now();
    let cakes = clam::CAKES::new(&space);
    let cakes = cakes.build(&partition_criteria);
    let build_time = start.elapsed().as_secs_f64();
    log::info!("Built tree to a depth of {} ...", cakes.depth());

    log::info!("Writing tree report on {}-{} data ...", data_name, metric_name);
    let (tree, clusters) = cakes.root().report_tree(build_time);
    let cluster_dir = output_dir.join("clusters");
    utils::make_dir(&cluster_dir, true)?;
    clusters
        .into_par_iter()
        .map(|report| utils::write_report(&report, &cluster_dir.join(format!("{}.json", report.name))))
        .collect::<Result<Vec<_>, String>>()?;

    log::info!("Starting search on {}-{} data ...", data_name, metric_name);

    let (reports, times): (Vec<_>, Vec<_>) = queries_radii
        .iter()
        .enumerate()
        .map(|(i, (query, radius))| {
            if (i + 1) % 10 == 0 {
                log::info!(
                    "Progress {:6.2}% on {}-{} data ...",
                    100. * (i as f64 + 1.) / queries_radii.len() as f64,
                    data_name,
                    metric_name
                );
            }
            let sample = (0..num_runs)
                .map(|_| {
                    let start = std::time::Instant::now();
                    let results = cakes.rnn_search(query, *radius);
                    (results, start.elapsed().as_secs_f64())
                })
                .collect::<Vec<_>>();
            let report = sample.first().unwrap().0.clone();
            let times = sample.into_iter().map(|(_, t)| t).collect::<Vec<_>>();
            (report, times)
        })
        .unzip();

    log::info!("Collecting report on {}-{} data ...", data_name, metric_name);

    let true_hits = neighbors.into_iter().map(|n_row| n_row.into_iter().collect());
    let hits: Vec<HashSet<usize>> = reports
        .iter()
        .map(|r| HashSet::from_iter(r.hits.iter().map(|v| *v)))
        .collect();
    let recalls: Vec<f64> = hits
        .iter()
        .zip(true_hits)
        .map(|(pred, actual)| {
            let intersection = pred.intersection(&actual).count();
            (intersection as f64) / (actual.len() as f64)
        })
        .collect();

    let report = utils::RnnReport {
        tree,
        radii: queries_radii.iter().map(|(_, r)| r.as_f64()).collect(),
        report: utils::BatchReport {
            num_queries: queries_radii.len(),
            num_runs,
            times,
            reports,
            recalls,
        },
    };

    let failures = report.validate();
    if failures.is_empty() {
        utils::write_report(report, &output_dir.join("rnn-report.json"))
    } else {
        Err(format!("The report was invalid:\n{:?}", failures.join("\n")))
    }
}

fn main() -> Result<(), String> {
    env_logger::Builder::new().parse_filters("info").init();

    let results = [
        search::<f32, f32, i32, f32, f32>("deep-image", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("fashion-mnist", "euclidean", 10),
        search::<f32, f32, i32, f32, f32>("gist", "euclidean", 10),
        search::<f32, f32, i32, f32, f32>("glove-25", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("glove-50", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("glove-100", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("glove-200", "cosine", 10),
        search::<f32, f64, i32, f32, f32>("lastfm", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("mnist", "euclidean", 10),
        search::<f32, f32, i32, f32, f32>("nytimes", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("sift", "euclidean", 10),
        // search::<bool, bool, i32, f32, u8>("kosarak", "jaccard", 10),
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
