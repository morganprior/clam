#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![allow(unused_imports)]

//! Tree export to CSV for creating lfd plots.

use core::cmp::Ordering;
use std::{path::Path, time::Instant};

use abd_clam::{Cakes, Dataset, PartitionCriteria, Tree, VecDataset};
use clap::Parser;
use distances::Number;
use log::info;
use num_format::ToFormattedString;
use serde::{Deserialize, Serialize};

mod ann_datasets;

use ann_datasets::AnnDatasets;

fn main() -> Result<(), String> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    // Check that the data set exists.
    let data_paths = [
        args.input_dir.join(format!("{}-train.npy", args.dataset)),
        args.input_dir.join(format!("{}-test.npy", args.dataset)),
    ];
    for path in &data_paths {
        if !path.exists() {
            return Err(format!("File {path:?} does not exist."));
        }
    }

    // Check that the output directory exists.
    if !args.output_dir.exists() {
        return Err(format!(
            "Output directory {:?} does not exist.",
            args.output_dir
        ));
    }

    info!("Starting data set {}.", args.dataset);

    export_tree(&args)?;

    Ok(())
}

/// Command line arguments for the replicating the ANN-Benchmarks results for Cakes.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory with the data sets. The directory should contain
    /// the hdf5 files downloaded from the ann-benchmarks repository.
    #[arg(long)]
    input_dir: std::path::PathBuf,
    /// Output directory for the report.
    #[arg(long)]
    output_dir: std::path::PathBuf,
    /// Name of the data set to process. `data_dir` should contain two files
    /// named `{name}-train.npy` and `{name}-test.npy`. The train file
    /// contains the data to be indexed for search, and the test file contains
    /// the queries to be searched for.
    #[arg(long)]
    dataset: String,
    /// Seed for the random number generator.
    #[arg(long)]
    seed: Option<u64>,
}

/// Export the clusters in the tree to CSV for creating lfd plots.
fn export_tree(args: &Args) -> Result<(), String> {
    let data_name = args.dataset.clone();
    let dataset = AnnDatasets::from_str(&data_name)?;
    let metric_name = dataset.metric_name();
    let metric = dataset.metric()?;
    let is_expensive = false;

    info!("Reading data set {data_name}.");

    let [data, _] = dataset.read(&args.input_dir)?;
    let data = VecDataset::new(data_name.clone(), data, metric, is_expensive);

    info!("Building tree for data set {data_name}.");
    let tree = Tree::new(data, args.seed)
        .partition(&PartitionCriteria::default())
        .with_ratios(true);

    info!("Exporting tree for data set {data_name}.");
    let tree_dir = args.output_dir.join(format!("{data_name}_{metric_name}"));
    std::fs::create_dir_all(&tree_dir).map_err(|e| e.to_string())?;

    let csv_path = tree_dir.join("clusters.csv");
    let mut csv_file = std::fs::File::create(&csv_path).map_err(|e| e.to_string())?;
    let mut csv_writer = csv::Writer::from_writer(&mut csv_file);
    csv_writer
        .write_record([
            "offset",
            "cardinality",
            "depth",
            "radius",
            "lfd",
            "ratio_cardinality",
            "ratio_radius",
            "ratio_lfd",
            "ratio_cardinality_ema",
            "ratio_radius_ema",
            "ratio_lfd_ema",
        ])
        .map_err(|e| e.to_string())?;

    for (i, &c) in tree.root().subtree().iter().enumerate() {
        let mut row = vec![
            c.offset().to_string(),
            c.cardinality().to_string(),
            c.depth().to_string(),
            c.radius().as_f64().to_string(),
            c.lfd().as_f64().to_string(),
        ];
        for r in c
            .ratios()
            .unwrap_or_else(|| unreachable!("We built the tree with ratios."))
        {
            row.push(r.to_string());
        }
        csv_writer.write_record(&row).map_err(|e| e.to_string())?;

        if i % 1000 == 0 {
            info!("Exported {} clusters.", i);
        }
    }

    csv_writer.flush().map_err(|e| e.to_string())?;

    Ok(())
}
