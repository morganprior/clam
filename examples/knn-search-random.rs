use abd_clam::cakes::CAKES;
use abd_clam::cluster::PartitionCriteria;
use abd_clam::dataset::VecVec;
use abd_clam::distances;
use abd_clam::utils::helpers;

fn cakes() {
    let data_name = "knn_f32_euclidean".to_string();
    let data: Vec<Vec<f32>> = helpers::gen_data_f32(2_000, 30, 0., 1., 42);
    let data = VecVec::new(data, distances::f32::euclidean, data_name, false);

    let num_queries = 10;
    let queries = helpers::gen_data_f32(num_queries, 100, 0., 1., 42);
    let criteria: PartitionCriteria<f32, _, VecVec<f32, _>> = PartitionCriteria::new(true).with_min_cardinality(1);

    let cakes = CAKES::new(data, Some(42)).build(&criteria);

    let queries = (0..num_queries).map(|i| &queries[i]).collect::<Vec<_>>();

    for k in [1, 10, 100] {
        let thresholds_nn = cakes.batch_knn_search(&queries, k);
        let actual_nn = cakes.batch_linear_search_knn(&queries, k);

        assert_eq!(thresholds_nn.len(), actual_nn.len());
        assert_eq!(thresholds_nn, actual_nn);
    }
}

fn main() {
    cakes()
}
