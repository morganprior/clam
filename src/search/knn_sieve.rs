use std::cmp::Ordering;

use crate::{prelude::*, utils::reports};

use super::find_kth;

#[derive(Debug)]
#[allow(dead_code)]
pub enum Delta {
    Center,
    Max,
    Min,
}

#[derive(Debug, Clone)]
struct Item<'a, T: Number, U: Number> {
    c: &'a Cluster<'a, T, U>,
    d: U,
}

impl<'a, T: Number, U: Number> PartialEq for Item<'a, T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.d == other.d
    }
}

impl<'a, T: Number, U: Number> PartialOrd for Item<'a, T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.d.partial_cmp(&other.d)
    }
}

struct OrdNumber<U: Number> {
    number: U,
}

impl<U: Number> PartialEq for OrdNumber<U> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl<U: Number> Eq for OrdNumber<U> {}

impl<U: Number> PartialOrd for OrdNumber<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl<U: Number> Ord for OrdNumber<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug)]
//See note above sieve_swap about choice to have 3 separate delta members

/// Struct for facilitating knn tree search
/// `clusters` represents the list of candidate Clusters
/// The ith element of `cumulative_cardinalities` represents the sum of cardinalities of the 0th through ith Cluster in
/// `clusters`.
pub struct KnnSieve<'a, T: Number, U: Number> {
    pub clusters: Vec<&'a Cluster<'a, T, U>>,
    pub query: &'a [T],
    pub k: usize,
    pub cumulative_cardinalities: Vec<usize>,
    pub deltas: Vec<U>,
    pub deltas_max: Vec<U>,
    pub deltas_min: Vec<U>,
}

impl<'a, T: Number, U: Number> KnnSieve<'a, T, U> {
    pub fn new(clusters: Vec<&'a Cluster<'a, T, U>>, query: &'a [T], k: usize) -> Self {
        KnnSieve {
            clusters,
            query,
            k,
            cumulative_cardinalities: vec![],
            deltas: vec![],
            deltas_max: vec![],
            deltas_min: vec![],
        }
    }

    pub fn build(mut self) -> Self {
        self.update_cumulative_cardinalities();

        self.deltas = self.clusters.iter().map(|&c| self.delta(c)).collect();

        (self.deltas_max, self.deltas_min) = self
            .clusters
            .iter()
            .zip(self.deltas.iter())
            .map(|(&c, d0)| (self.delta_max(c, *d0), self.delta_min(c, *d0)))
            .unzip();

        self
    }

    pub fn append_history(&self, report: &mut reports::SearchReport) {
        let clusters = self
            .clusters
            .iter()
            .map(|c| c.name_str())
            .zip(self.deltas.iter().map(|d| d.as_f64()));
        report.history.extend(clusters);
    }

    fn delta(&self, c: &Cluster<T, U>) -> U {
        c.distance_to_instance(self.query)
    }

    fn delta_max(&self, c: &Cluster<T, U>, d0: U) -> U {
        d0 + c.radius()
    }

    fn delta_min(&self, c: &Cluster<T, U>, d0: U) -> U {
        if d0 > c.radius() {
            d0 - c.radius()
        } else {
            U::zero()
        }
    }

    fn find_kth(&mut self, delta: &Delta) -> U {
        let deltas = match delta {
            Delta::Center => self.deltas.clone(),
            Delta::Max => self.deltas_max.clone(),
            Delta::Min => self.deltas_min.clone(),
        };
        let items = self
            .clusters
            .iter()
            .zip(deltas.into_iter())
            .map(|(&c, d)| Item { c, d })
            .collect::<Vec<_>>();

        let other_deltas = {
            let (d1s, d2s) = match delta {
                Delta::Center => (self.deltas_max.clone(), self.deltas_min.clone()),
                Delta::Max => (self.deltas.clone(), self.deltas_min.clone()),
                Delta::Min => (self.deltas.clone(), self.deltas_max.clone()),
            };
            d1s.into_iter().zip(d2s.into_iter()).collect::<Vec<_>>()
        };

        let (items, other_deltas, cumulative_cardinalities) =
            find_kth::find_kth(&items, &self.cumulative_cardinalities, &other_deltas, self.k);

        let (clusters, deltas): (Vec<_>, Vec<_>) = items.into_iter().map(|i| (i.c, i.d)).unzip();

        let (d1s, d2s): (Vec<_>, Vec<_>) = other_deltas.into_iter().unzip();

        self.clusters = clusters;
        self.cumulative_cardinalities = cumulative_cardinalities;
        let index = self
            .cumulative_cardinalities
            .iter()
            .enumerate()
            .find(|(_, &c)| c > self.k)
            .unwrap()
            .0;
        match delta {
            Delta::Center => {
                self.deltas = deltas;
                self.deltas_max = d1s;
                self.deltas_min = d2s;
                self.deltas[index]
            }
            Delta::Max => {
                self.deltas = d1s;
                self.deltas_max = deltas;
                self.deltas_min = d2s;
                self.deltas_max[index]
            }
            Delta::Min => {
                self.deltas = d1s;
                self.deltas_max = d2s;
                self.deltas_min = deltas;
                self.deltas_min[index]
            }
        }
    }

    /// Gets the value of the given delta type (0, 1, or 2) based on index (of Cluster)
    ///
    /// `delta0` = dist from query to Cluster center
    /// `delta1` = dist from query to potentially farthest instance in Cluster
    /// `delta2` = dist from query to potentially closest instance in Cluster
    #[allow(dead_code)]
    fn get_delta_by_cluster_index(&self, index: usize, delta: &Delta) -> U {
        match delta {
            Delta::Center => self.deltas[index],
            Delta::Max => self.deltas_max[index],
            Delta::Min => self.deltas_min[index],
        }
    }

    pub fn replace_with_child_clusters(mut self) -> Self {
        let (clusters, deltas_0, deltas_1, deltas_2) = self
            .clusters
            .iter()
            .zip(self.deltas.iter())
            .zip(self.deltas_max.iter())
            .zip(self.deltas_min.iter())
            .fold(
                (vec![], vec![], vec![], vec![]),
                |(mut c, mut d0, mut d1, mut d2), (((&c_, &d0_), &d1_), &d2_)| {
                    if c_.is_leaf() {
                        c.push(c_);
                        d0.push(d0_);
                        d1.push(d1_);
                        d2.push(d2_);
                    } else {
                        let [l, r] = c_.children();
                        c.extend_from_slice(&[l, r]);
                        let l0 = self.delta(l);
                        let r0 = self.delta(l);
                        d0.extend_from_slice(&[l0, r0]);
                        d1.extend_from_slice(&[self.delta_max(l, l0), self.delta_max(r, r0)]);
                        d2.extend_from_slice(&[self.delta_min(l, l0), self.delta_min(r, r0)]);
                    }
                    (c, d0, d1, d2)
                },
            );

        self.clusters = clusters;
        self.deltas = deltas_0;
        self.deltas_max = deltas_1;
        self.deltas_min = deltas_2;
        self.update_cumulative_cardinalities();

        self
    }

    pub fn filter(mut self) -> Self {
        let d1_k = self.find_kth(&Delta::Max);
        let keep = self.deltas_min.iter().map(|d2| *d2 <= d1_k);

        (self.clusters, self.deltas, self.deltas_max, self.deltas_min) = self
            .clusters
            .into_iter()
            .zip(self.deltas.into_iter())
            .zip(self.deltas_max.into_iter())
            .zip(self.deltas_min.clone().into_iter())
            .zip(keep)
            .fold(
                (vec![], vec![], vec![], vec![]),
                |(mut c, mut d0, mut d1, mut d2), ((((c_, d0_), d1_), d2_), k_)| {
                    if k_ {
                        c.push(c_);
                        d0.push(d0_);
                        d1.push(d1_);
                        d2.push(d2_);
                    }
                    (c, d0, d1, d2)
                },
            );

        self.update_cumulative_cardinalities();
        self
    }

    fn update_cumulative_cardinalities(&mut self) {
        // TODO: Take l, r indices and onlt update in between them
        self.cumulative_cardinalities = self
            .clusters
            .iter()
            .scan(0, |acc, c| {
                *acc += c.cardinality();
                Some(*acc)
            })
            .collect();
    }

    pub fn are_all_leaves(&self) -> bool {
        self.clusters.iter().all(|c| c.is_leaf())
    }

    /// Returns `k` best hits from the sieve along with their distances from the
    /// query. If this method is called when the `are_all_leaves` method
    /// evaluates to `true`, the result will have the best recall. If the
    /// `metric` in use obeys the triangle inequality, then the results will
    /// have perfect recall. If this method is called before the sieve has been
    /// filtered down to the leaves, the results may not have perfect recall.
    pub fn extract(&self) -> (Vec<usize>, Vec<U>) {
        let mut candidates = self
            .clusters
            .iter()
            .cloned()
            .zip(self.deltas_min.iter().cloned())
            .collect::<Vec<_>>();

        let mut pq = priority_queue::DoublePriorityQueue::new();
        let space = self.clusters.first().unwrap().space();

        while !candidates.is_empty() {
            let candidate = candidates.pop().unwrap().0;
            let indices = candidate.indices();
            let instances = indices.iter().map(|&i| space.data().get(i)).collect::<Vec<_>>(); // TODO: Avoid loading these
            let distances = space.metric().one_to_many(self.query, &instances);

            indices.into_iter().zip(distances.into_iter()).for_each(|(i, d)| {
                pq.push(i, OrdNumber { number: d });
            });

            let mut potential_ties = vec![];
            while pq.len() > self.k {
                potential_ties.push(pq.pop_max().unwrap());
            }

            let threshold = pq.peek_max().unwrap().1.number;
            pq.extend(potential_ties.into_iter().filter(|(_, d)| d.number == threshold));

            candidates = candidates.into_iter().filter(|(_, d)| *d <= threshold).collect();
        }

        pq.into_iter().map(|(i, d)| (i, d.number)).unzip()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn find_2nd_delta0() {
        let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
        let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
        let metric = metric_from_name::<f64, f64>("euclideansq", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref(), false);
        let partition_criteria = crate::PartitionCriteria::new(true)
            .with_max_depth(3)
            .with_min_cardinality(1);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);

        let flat_tree = cluster.subtree();
        let mut sieve = KnnSieve::new(flat_tree, &data[0], 2).build();

        assert_eq!(sieve.find_kth(&Delta::Center), 3.);
    }
}
