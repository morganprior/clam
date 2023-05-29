use std::f64::EPSILON;

use crate::core::number::Number;

pub fn cosine<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let [xx, yy, xy] = x.iter().zip(y.iter()).fold([T::zero(); 3], |[xx, yy, xy], (&a, &b)| {
        [xx + a * a, yy + b * b, xy + a * b]
    });
    let [xx, yy, xy] = [xx.as_f64(), yy.as_f64(), xy.as_f64()];

    if xx <= EPSILON || yy <= EPSILON || xy <= EPSILON {
        U::one()
    } else {
        let d = 1. - xy / (xx * yy).sqrt();
        if d < EPSILON {
            U::zero()
        } else {
            U::from(d).unwrap()
        }
    }
}

pub fn hamming<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    U::from(x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count()).unwrap()
}


pub fn levenshtein<T: Number, U: Number>(x: &[T], y: &[T]) -> U {
    let (len_x, len_y) = (x.iter().count(), y.iter().count());

    if len_x == 0 {
        // handle special case of 0 length
        U::from(len_x).unwrap()
    } else if len_y == 0 {
        // handle special case of 0 length
        U::from(len_y).unwrap()
    } else if len_x < len_y {
        // require len_a < len_b
        levenshtein(y, x)
    } else {
        let len_y = len_y + 1;

        // initialize DP table for string y
        let mut cur: Vec<usize> = (0..len_y).collect();

        // calculate edit distance
        for (i, cx) in x.iter().enumerate() {
            // get first column for this row
            let mut pre = cur[0];
            cur[0] = i + 1;
            for (j, cy) in y.iter().enumerate() {
                let tmp = cur[j + 1];
                cur[j + 1] = std::cmp::min(
                    // deletion
                    tmp + 1,
                    std::cmp::min(
                        // insertion
                        cur[j] + 1,
                        // match or substitution
                        pre + if cx == cy { 0 } else { 1 },
                    ),
                );
                pre = tmp;
            }
        }
        U::from(cur[len_y - 1]).unwrap()
    }
}


#[cfg(test)]
mod tests {
    use crate::distances::misc::levenshtein; 


    #[test]
    fn test_levenshtein() {
        let x = [0, 1, 2, 2, 1, 3, 4]; 
        let y = [5, 1, 2, 2, 6, 3]; 
        
        let lev: i32 = levenshtein(&x, &y); 

        assert_eq!(lev, 3)
        
        }
    }
