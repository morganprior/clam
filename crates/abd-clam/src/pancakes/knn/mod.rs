//! K-Nearest Neighbors search in a compressed space.

mod breadth_first_sieve;
mod depth_first_sieve;
mod linear;

use distances::number::UInt;

use crate::Instance;

use super::CodecData;

/// The algorithm to use for K-Nearest Neighbors search.
pub enum Algorithm {
    /// Use linear search on the dataset.
    Linear,
    /// Search the dataset using the Depth-First Sieve algorithm.
    DepthFirstSieve,
    /// Search the dataset using the Breadth-First Sieve algorithm.
    BreadthFirstSieve,
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::Linear
    }
}

impl Algorithm {
    /// Searches for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `tree` - The tree to search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn search<I, U, M>(&self, query: &I, k: usize, data: &CodecData<I, U, M>) -> Vec<(usize, U)>
    where
        I: Instance,
        U: UInt,
        M: Instance,
    {
        match self {
            Self::Linear => linear::search(query, k, data),
            Self::DepthFirstSieve => depth_first_sieve::search(query, k, data),
            Self::BreadthFirstSieve => breadth_first_sieve::search(query, k, data),
        }
    }

    /// Returns the name of the algorithm.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Linear => "Linear",
            Self::DepthFirstSieve => "DepthFirstSieve",
            Self::BreadthFirstSieve => "BreadthFirstSieve",
        }
    }

    /// Returns the algorithm from a string representation of the name.
    ///
    /// The string is case-insensitive.
    ///
    /// # Arguments
    ///
    /// * `s` - The string representation of the algorithm.
    ///
    /// # Returns
    ///
    /// The algorithm variant.
    ///
    /// # Errors
    ///
    /// If the string does not match any of the algorithms.
    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "depthfirstsieve" => Ok(Self::DepthFirstSieve),
            "breadthfirstsieve" => Ok(Self::BreadthFirstSieve),
            _ => Err(format!("Unknown algorithm: {s}")),
        }
    }

    /// Returns the baseline algorithm, which is Linear
    #[must_use]
    pub const fn baseline() -> Self {
        Self::Linear
    }
}
