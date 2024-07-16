use std::io::Write;
use std::path::Path;

use rand::prelude::*;

use abd_clam::{
    pancakes::{decode_general, encode_general, knn, rnn, CodecData, SquishyBall},
    Cakes, Cluster, Dataset, PartitionCriteria, VecDataset,
};
use distances::{strings::Penalties, Number};
use symagen::random_edits::{are_we_there_yet, generate_clumped_data, generate_random_string};

#[allow(clippy::ptr_arg)]
fn lev_metric(x: &String, y: &String) -> u16 {
    distances::strings::levenshtein(x, y)
}

/// Writes the given clumped data to a .txt file.
///
/// The file name will be "n-m.txt" where n is the number of clumps and m is the clump size.
/// The file will contain pairs of lines, with the first line being the metadata for the point
/// and the second line being the point itself.
/// The metadata lines are of the form ixj where i is the clump index and j is the index of the point within the clump.
///
/// # Arguments
///
/// `clumped_data`: The clumped data to write to the file.
/// `path`: The directory where the file will be created.
///
/// # Errors
///
/// * If the path does not exist
/// * If the path is not a directory
/// * If `clumped_data` is empty
/// * If the file cannot be created
fn write_data(clumped_data: &[(String, String)], path: &Path) -> Result<(), String> {
    // Check if path exits
    if !path.exists() {
        return Err("Path does not exist".to_string());
    }

    // Check if the path is a directory
    if !path.is_dir() {
        return Err("Path is not a directory".to_string());
    }

    // Check if the clumped data is empty
    if clumped_data.is_empty() {
        return Err("No data to write".to_string());
    }

    let (n, m) = clumped_data
        .last()
        .map(|(x, _)| {
            let mut parts = x.split('x');
            let n = parts.next().unwrap().parse::<usize>().unwrap();
            let m = parts.next().unwrap().parse::<usize>().unwrap();
            (n + 1, m + 1)
        })
        .ok_or("No data to write")?;

    let filename = format!("{n}-{m}.txt");

    let filepath = path.join(filename);

    let mut file = std::fs::File::create(filepath).map_err(|e| e.to_string())?;
    for (x, y) in clumped_data {
        writeln!(file, "{x}").map_err(|e| e.to_string())?;
        writeln!(file, "{y}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn read_data(path: &Path) -> Result<Vec<(String, String)>, String> {
    let lines = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let lines = lines.lines().map(|x| x.to_string()).collect::<Vec<_>>();

    Ok(lines
        .chunks(2)
        .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
        .collect::<Vec<_>>())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 2 {
        return Err(format!(
            "Expected single input parameter, the directory where dataset will be saved. Got {args:?} instead",
        )
        );
    }

    // Parse args[0] into path object
    let dataset_dir = Path::new(&args[1]);
    let dataset_dir = std::fs::canonicalize(dataset_dir).map_err(|e| e.to_string())?;

    // Check that `dataset_dir` has a parent
    if dataset_dir.parent().is_none() {
        return Err("No parent directory".to_string());
    }

    // Check whether the parent of dataset_dir exists
    if !dataset_dir.parent().unwrap().exists() {
        return Err(format!("Parent of {dataset_dir:?} does not exist."));
    }

    // If dataset_dir does not exist, create it.
    if !dataset_dir.exists() {
        std::fs::create_dir(&dataset_dir).map_err(|e| e.to_string())?;
    }

    let seed = 42;
    let clump_rng = &mut rand::rngs::StdRng::seed_from_u64(seed);

    let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect::<Vec<_>>();
    // let seed_string = generate_random_string(100, &alphabet, &mut rand::rngs::StdRng::seed_from_u64(seed));
    let seed_string = generate_random_string(100, &alphabet);
    let penalties = Penalties::<u16>::new(0, 1, 1);

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
    for &(n, m) in &sizes[..1] {
        println!();

        let expected_name = format!("{n}-{m}.txt");
        println!("Dataset: {expected_name}");

        let expected_path = dataset_dir.join(&expected_name);
        // let inter_clump_distance: i32 = 15;
        let clumped_data = if !expected_path.exists() {
            let clumped_data =
                generate_clumped_data(&seed_string, penalties, &alphabet, n, m, 5, seed);
            write_data(&clumped_data, &dataset_dir)?;
            clumped_data
        } else {
            read_data(&expected_path)?
        };

        // Select 10 random queries from the clumped data
        let (queries, clumped_data) = {
            let mut clumped_data = clumped_data;
            // Shuffle the clumped data
            let mut rng: ThreadRng = rand::thread_rng();
            clumped_data.shuffle(&mut rng);
            let queries = clumped_data
                .iter()
                .take(1) // number of queries to take
                .map(|(m, p)| {
                    let q = are_we_there_yet(p, penalties, 5, &alphabet, clump_rng);
                    (m.clone(), q)
                })
                .collect::<Vec<_>>();
            (queries, clumped_data)
        };

        let (query_meta, query_data): (Vec<_>, Vec<_>) = queries.into_iter().unzip();
        let (clumped_meta, clumped_data) = clumped_data.into_iter().unzip();
        let radius = 5;
        let k = 5;

        let name = format!("{n}x{m}");
        let dataset =
            VecDataset::new(name, clumped_data, lev_metric, true).assign_metadata(clumped_meta)?;

        // Get a baseline for linear rnn search

        let (baseline_hits_rnn, baseline_time_rnn) =
            baseline_rnn_search(&dataset, &query_data, radius);
        check_baseline_metadata(&query_meta, &baseline_hits_rnn);

        // Get a baseline for linear knn search

        let (hits_knn_baseline, time_knn_baseline) = baseline_knn_search(&dataset, &query_data, k);
        check_baseline_metadata(&query_meta, &hits_knn_baseline);

        let cakes = build_cakes_tree(dataset);

        // Run clustered rnn search

        let cakes_rnn = cakes_rnn_search(&cakes, &query_data, radius);
        println!(
            "RNN ratio: baseline / cakes = {:.4}",
            baseline_time_rnn / cakes_rnn
        );

        // Run Cakes knn search (specify algorithm)
        let cakes_knn = cakes_knn_search(
            &cakes,
            &query_data,
            k,
            abd_clam::cakes::knn::Algorithm::Linear,
        );
        println!(
            "KNN ratio: baseline / cakes = {:.4}",
            time_knn_baseline / cakes_knn
        );

        // Create squishy tree from cakes tree
        let cakes_tree = cakes.trees()[0];
        let dataset = cakes_tree.data();
        let root = cakes_tree.root().clone();
        let tree_size = root.subtree().len();
        let root = SquishyBall::from_base_tree(root, dataset);

        let compression_time = std::time::Instant::now();
        let metadata = dataset.metadata().to_vec();
        let dataset = CodecData::new(
            root,
            dataset,
            encode_general::<u16>,
            decode_general,
            metadata,
        )?;

        // Write the dataset to a binary file
        let bin_dir = dataset_dir.join(format!("codec-{n}-{m}"));
        dataset.save(&bin_dir)?;
        let compression_time = compression_time.elapsed().as_secs_f32();
        println!("Dataset compressed in {compression_time:.4}s");

        // Reload the dataset and tree, and check that they are the same
        let decompression_time = std::time::Instant::now();
        let re_data = CodecData::<String, u16, String>::load(
            &bin_dir,
            lev_metric,
            true,
            encode_general::<u16>,
            decode_general,
        )?;
        let decompression_time = decompression_time.elapsed().as_secs_f32();
        println!("Dataset decompressed in {decompression_time:.4}s");

        confirm_successful_reload(&re_data, &dataset);

        // Run rnn with the queries on the reloaded dataset
        let (codec_hits_rnn, codec_time_rnn) = codec_rnn_search(&re_data, &query_data, radius);

        // Check that the hits are the same
        check_hits(&query_meta, &codec_hits_rnn, &baseline_hits_rnn);

        println!(
            "Codec RNN: {codec_time_rnn:.4}s, num_queries: {}",
            codec_hits_rnn.len()
        );
        println!(
            "RNN ratio: baseline / codec = {:.4}",
            baseline_time_rnn / codec_time_rnn
        );

        // Run knn with the queries on the reloaded dataset
        let (codec_hits_knn, codec_time_knn) =
            codec_knn_search(&re_data, &query_data, k, knn::Algorithm::Linear);

        // Check that the hits are the same
        check_hits(&query_meta, &codec_hits_knn, &hits_knn_baseline);

        println!(
            "Codec KNN: {codec_time_knn:.4}s, num_queries: {}",
            codec_hits_knn.len()
        );
        println!(
            "KNN ratio: baseline / codec = {:.4}",
            time_knn_baseline / codec_time_knn
        );

        // Get the size of the binary files in the codec directory
        let bin_size = bin_dir
            .read_dir()
            .map_err(|e| e.to_string())?
            .map(|entry| entry.unwrap().metadata().unwrap().len())
            .sum::<u64>();

        // Get the size of the text file
        let txt_size = expected_path.metadata().map_err(|e| e.to_string())?.len();

        let trimmed_size = dataset.root().compressible_subtree().len();

        println!(
            "Root: {}, Clusters: {tree_size}, Trimmed: {trimmed_size}",
            dataset.root().name()
        );

        println!(
            "Recursive cost: {}, Unitary cost: {}, Estimated Factor: {:.2e}",
            dataset.root().recursive_cost(),
            dataset.root().unitary_cost(),
            dataset.root().recursive_cost().as_f64() / dataset.root().unitary_cost().as_f64()
        );

        println!(
            "File sizes: Text: {txt_size}, Binary: {bin_size}, Actual Factor: {:.2e}",
            bin_size.as_f64() / txt_size.as_f64()
        );
    }

    Ok(())
}

fn codec_knn_search(
    re_data: &CodecData<String, u16, String>,
    query_data: &[String],
    k: usize,
    algorithm: knn::Algorithm,
) -> (Vec<Vec<(String, u16)>>, f32) {
    println!("Running KNN search");
    let codec_knn = std::time::Instant::now();
    let codec_hits_knn = query_data
        .iter()
        .map(|q| re_data.knn_search(q, k, &algorithm))
        .collect::<Vec<_>>();
    let codec_knn = codec_knn.elapsed().as_secs_f32();
    let codec_hits_knn = codec_hits_knn
        .into_iter()
        .map(|h| {
            h.into_iter()
                .map(|(i, d)| (re_data.metadata()[i].clone(), d))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    (codec_hits_knn, codec_knn)
}

fn codec_rnn_search(
    re_data: &CodecData<String, u16, String>,
    query_data: &[String],
    radius: u16,
) -> (Vec<Vec<(String, u16)>>, f32) {
    println!("Running RNN search");
    let codec_rnn = std::time::Instant::now();
    let codec_hits_rnn = query_data
        .iter()
        .map(|q| re_data.rnn_search(q, radius, &rnn::Algorithm::Clustered))
        .collect::<Vec<_>>();
    let codec_rnn = codec_rnn.elapsed().as_secs_f32();
    let codec_hits_rnn = codec_hits_rnn
        .into_iter()
        .map(|h| {
            h.into_iter()
                .map(|(i, d)| (re_data.metadata()[i].clone(), d))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    (codec_hits_rnn, codec_rnn)
}

fn cakes_knn_search(
    cakes: &Cakes<String, u16, VecDataset<String, u16, String>>,
    query_data: &[String],
    k: usize,
    algorithm: abd_clam::cakes::knn::Algorithm,
) -> f32 {
    let cakes_knn = std::time::Instant::now();
    let cakes_hits_knn = query_data
        .iter()
        .map(|q| cakes.knn_search(q, k, algorithm))
        .collect::<Vec<_>>();
    let cakes_knn = cakes_knn.elapsed().as_secs_f32();
    println!(
        "Cakes KNN: {cakes_knn:.4}s, num_queries: {}",
        cakes_hits_knn.len()
    );

    cakes_knn
}

fn cakes_rnn_search(
    cakes: &Cakes<String, u16, VecDataset<String, u16, String>>,
    query_data: &[String],
    radius: u16,
) -> f32 {
    let cakes_rnn = std::time::Instant::now();
    let cakes_hits_rnn = query_data
        .iter()
        .map(|q| cakes.rnn_search(q, radius, abd_clam::cakes::rnn::Algorithm::Clustered))
        .collect::<Vec<_>>();
    let cakes_rnn = cakes_rnn.elapsed().as_secs_f32();
    println!(
        "Cakes RNN: {cakes_rnn:.4}s, num_queries: {}",
        cakes_hits_rnn.len()
    );

    cakes_rnn
}

fn baseline_rnn_search(
    dataset: &VecDataset<String, u16, String>,
    query_data: &[String],
    radius: u16,
) -> (Vec<Vec<(String, u16)>>, f32) {
    let baseline_rnn = std::time::Instant::now();
    let hits_rnn = query_data
        .iter()
        .map(|q| dataset.par_linear_rnn(q, radius))
        .collect::<Vec<_>>();
    let baseline_rnn = baseline_rnn.elapsed().as_secs_f32();
    let hits_rnn_baseline = hits_rnn
        .into_iter()
        .map(|h| {
            h.into_iter()
                .map(|(i, d)| (dataset.metadata_of(i).clone(), d))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    println!(
        "Baseline RNN: {baseline_rnn:.4}s, num_queries: {}",
        hits_rnn_baseline.len()
    );
    (hits_rnn_baseline, baseline_rnn)
}

fn baseline_knn_search(
    dataset: &VecDataset<String, u16, String>,
    query_data: &[String],
    k: usize,
) -> (Vec<Vec<(String, u16)>>, f32) {
    let baseline_knn = std::time::Instant::now();
    let hits_knn = query_data
        .iter()
        .map(|q| dataset.par_linear_knn(q, k))
        .collect::<Vec<_>>();
    let baseline_knn = baseline_knn.elapsed().as_secs_f32();
    let hits_knn_baseline = hits_knn
        .into_iter()
        .map(|h| {
            h.into_iter()
                .map(|(i, d)| (dataset.metadata_of(i).clone(), d))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    println!(
        "Baseline KNN: {baseline_knn:.4}s, num_queries: {}",
        hits_knn_baseline.len()
    );

    (hits_knn_baseline, baseline_knn)
}

fn check_baseline_metadata(query_meta: &[String], hits: &[Vec<(String, u16)>]) {
    for (qm, hits) in query_meta.iter().zip(hits.iter()) {
        // Check that the metadata matches
        assert!(hits.iter().any(|(m, _)| m == qm))
    }
}

fn build_cakes_tree(
    dataset: VecDataset<String, u16, String>,
) -> Cakes<String, u16, VecDataset<String, u16, String>> {
    // Build Cakes tree
    let criteria = PartitionCriteria::default();
    let seed = Some(42);
    let cakes_time = std::time::Instant::now();
    let cakes = Cakes::new(dataset, seed, &criteria);
    let cakes_time = cakes_time.elapsed().as_secs_f32();
    println!("Cakes built in {cakes_time:.4}s");

    cakes
}

fn confirm_successful_reload(
    re_data: &CodecData<String, u16, String>,
    dataset: &CodecData<String, u16, String>,
) {
    assert_eq!(dataset.root().subtree(), re_data.root().subtree());
    assert_eq!(dataset.centers(), re_data.centers());
    assert_eq!(dataset.metadata(), re_data.metadata());

    for (&c, &rc) in dataset
        .root()
        .subtree()
        .iter()
        .zip(re_data.root().subtree().iter())
    {
        assert_eq!(c, rc);
        assert_eq!(c.arg_center(), rc.arg_center());
        assert_eq!(c.arg_radial(), rc.arg_radial());
        assert_eq!(c.radius(), rc.radius());
        assert_eq!(c.arg_poles(), rc.arg_poles());
        assert_eq!(c.recursive_cost(), rc.recursive_cost());
        assert_eq!(c.unitary_cost(), rc.unitary_cost());
    }
}

fn check_hits(
    query_meta: &[String],
    hits_pred: &[Vec<(String, u16)>],
    hits_true: &[Vec<(String, u16)>],
) {
    for ((qm, h_pred), h_true) in query_meta
        .iter()
        .zip(hits_pred.iter())
        .zip(hits_true.iter())
    {
        // Check that the metadata matches
        assert!(h_pred.iter().any(|(m, _)| m == qm));

        // Check that the number of hits is the same
        assert_eq!(h_pred.len(), h_true.len());

        // Sort the hits by metadata
        let mut codec_hits = h_pred.clone();
        codec_hits.sort_by_key(|(_, d)| *d);
        let mut hits = h_true.clone();
        hits.sort_by_key(|(_, d)| *d);

        println!("Pred hits: {:?}", codec_hits);
        println!("True Hits: {:?}", hits);

        // Check that the hits are the same
        for ((m1, d1), (m2, d2)) in codec_hits.iter().zip(hits.iter()) {
            assert_eq!(d1, d2);
            let m1 = m1.split('x').next().unwrap();
            let m2 = m2.split('x').next().unwrap();
            assert_eq!(m1, m2);
        }
    }
}
