//! Generate random strings for use in benchmarks, tests, and compression experiments.
use rand::prelude::*;

use distances::{
    number::UInt,
    strings::{levenshtein_custom, needleman_wunsch::apply_edits, Edit, Penalties},
    Number,
};

/// Generates a random string of a given length from a given alphabet.
///
/// # Arguments
///
/// * `length`: The length of the string to generate.
/// * `alphabet`: The alphabet to choose characters from.
///
/// # Returns
///
/// A random string of the given length from the given alphabet.
#[must_use]
pub fn generate_random_string(length: usize, alphabet: &[char]) -> String {
    (0..length)
        .map(|_| alphabet[rand::random::<usize>() % alphabet.len()])
        .collect()
}

/// Generates (but does not apply) a random edit to a given string.
/// The character (if applicable) for the edit is a random character from the given alphabet.
///
/// # Arguments
///
/// * `string`: The string to apply the edit to.
/// * `alphabet`: The alphabet to choose the character from.
/// * `rng`: The random number generator.
///
/// # Returns
///
/// A random edit (Deletion, Insertion, or Substitution) based on the given string and alphabet.
#[must_use]
pub fn generate_random_edit<R: Rng>(string: &str, alphabet: &[char], rng: &mut R) -> Edit {
    let edit_type = u8::next_random(rng) % 3;
    let length = string.len();
    let char = alphabet[usize::next_random(rng) % alphabet.len()];

    match edit_type {
        0 => {
            let index = usize::next_random(rng) % (length + 1);
            Edit::Ins(index, char)
        }
        1 => Edit::Del(usize::next_random(rng) % length),
        2 => Edit::Sub(usize::next_random(rng) % length, char),
        _ => unreachable!(),
    }
}

/// Applies a random edit to a given string.
///
///
/// # Arguments
///
/// * `string`: The string to apply the edit to.
/// * `alphabet`: The alphabet to choose the character from.
/// * `rng`: The random number generator.
///
/// # Returns
///
/// A string with a random edit applied.
#[must_use]
pub fn apply_random_edit<R: Rng>(string: &str, alphabet: &[char], rng: &mut R) -> String {
    let random_edit = generate_random_edit(string, alphabet, rng);
    apply_edits(string, &[random_edit])
}

/// Randomly applies edits to a string until the distance between the new string and the seed string is greater than or equal to the target distance.
///
/// # Arguments
///
/// * `seed_string`: The string to start with.
/// * `penalties`: The penalties for the edit operations.
/// * `target_distance`: The desired distance between the seed and the new string.
/// * `alphabet`: The alphabet to choose characters from.
/// * `rng`: The random number generator.
///
/// # Returns
///
/// A randomly generated string with a distance greater than or equal to the target distance from the seed string.
pub fn are_we_there_yet<U: UInt, R: Rng>(
    seed_string: &str,
    penalties: Penalties<U>,
    target_distance: U,
    alphabet: &[char],
    rng: &mut R,
) -> String {
    let mut new_string = seed_string.to_string();
    let mut distance = U::zero();
    let lev = levenshtein_custom(penalties);

    while distance < target_distance {
        let edit = generate_random_edit(&new_string, alphabet, rng);
        new_string = apply_edits(&new_string, &[edit]);
        distance = lev(seed_string, &new_string);
    }

    new_string
}

/// Generates a batch of strings with random edits applied to a seed string.
///
/// # Arguments
///
/// * `seed_string`: The string to start with.
/// * `penalties`: The penalties for the edit operations.
/// * `target_distance`: The desired distance between the seed and the new string.
/// * `alphabet`: The alphabet to choose characters from.
/// * `batch_size`: The number of strings to generate.
/// * `rng`: The random number generator.
///
/// # Returns
///
/// A vector of randomly generated strings with a distance slightly greater than or equal to the target distance from the seed string.
pub fn create_batch<U: UInt, R: Rng>(
    seed_string: &str,
    penalties: Penalties<U>,
    min_distance: U,
    max_distance: U,
    alphabet: &[char],
    batch_size: usize,
    rng: &mut R,
) -> Vec<String> {
    let mut others = (1..batch_size)
        .map(|_| {
            // Randomly sample a distance between 1 and `target_distance`, inclusive
            let d = UInt::as_u64(min_distance)
                + u64::next_random(rng) % UInt::as_u64(max_distance - min_distance + U::one());
            are_we_there_yet(seed_string, penalties, U::from(d), alphabet, rng)
        })
        .collect::<Vec<_>>();
    others.push(seed_string.to_string());
    others
}

#[must_use]
/// Generates a random batch of strings in distinct clumps.
///
/// # Arguments
///
/// * `seed_string`: A baseline string to serve as the center of one clump.
/// * `penalties`: The penalties for the edit operations.
/// * `alphabet`: The alphabet to choose characters from.
/// * `num_clumps`: The number of clumps to generate.
/// * `clump_size`: The number of strings in each clump.
/// * `clump_radius`: The target distance from the seed string for each clump.
/// * `seed`: The seed for the random number generator.
///
/// # Returns
///
/// A vector of randomly generated strings in distinct clumps.
pub fn generate_clumped_data<U: UInt>(
    seed_string: &str,
    penalties: Penalties<U>,
    alphabet: &[char],
    num_clumps: usize,
    clump_size: usize,
    clump_radius: U,
    seed: u64,
) -> Vec<(String, String)> {
    // TODO(Morgan): add min length and max length for strings as inputs here

    // Vector of seed strings for each clump (can think of as the ``center''s of each clump)

    // TODO(Morgan): change 10 to input parameter inter-clump distance
    let rng = &mut rand::rngs::StdRng::seed_from_u64(seed);
    let min_distance = clump_radius * U::from(7);
    let max_distance = clump_radius * U::from(10);
    let clump_seeds = create_batch(
        seed_string,
        penalties,
        min_distance,
        max_distance,
        alphabet,
        num_clumps,
        rng,
    );

    // Generate the clumps
    clump_seeds
        .iter()
        .enumerate()
        .flat_map(|(i, seed_string)| {
            create_batch(
                seed_string,
                penalties,
                U::one(),
                clump_radius,
                alphabet,
                clump_size,
                rng,
            )
            .into_iter()
            .enumerate()
            .map(move |(j, string)| (format!("{i}x{j}"), string))
        })
        .collect()
}
