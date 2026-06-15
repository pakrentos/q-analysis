/// Fixed-width bitset over vertex ids; all sets in one complex share the same width.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    pub fn empty(num_bits: usize) -> Self {
        Bitset {
            words: vec![0; num_bits.div_ceil(64)],
        }
    }

    pub fn insert(&mut self, bit: u32) {
        self.words[(bit / 64) as usize] |= 1u64 << (bit % 64);
    }

    pub fn remove(&mut self, bit: u32) {
        self.words[(bit / 64) as usize] &= !(1u64 << (bit % 64));
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn intersection_count(&self, other: &Self) -> u32 {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a & b).count_ones())
            .sum()
    }

    pub fn intersection(&self, other: &Self) -> Self {
        Bitset {
            words: self
                .words
                .iter()
                .zip(&other.words)
                .map(|(a, b)| a & b)
                .collect(),
        }
    }

    /// Elements of `self` that are not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        Bitset {
            words: self
                .words
                .iter()
                .zip(&other.words)
                .map(|(a, b)| a & !b)
                .collect(),
        }
    }

    pub fn union_with(&mut self, other: &Self) {
        for (w, o) in self.words.iter_mut().zip(&other.words) {
            *w |= o;
        }
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, &word)| OneBits {
            word,
            base: (wi * 64) as u32,
        })
    }

    /// Orders equal-size sets as their sorted vertex lists: the set owning the
    /// lowest differing bit contains a smaller vertex the other lacks.
    pub fn lex_less(&self, other: &Self) -> bool {
        for (a, b) in self.words.iter().zip(&other.words) {
            if a != b {
                let lowest_diff = (a ^ b) & (a ^ b).wrapping_neg();
                return a & lowest_diff != 0;
            }
        }
        false
    }
}

struct OneBits {
    word: u64,
    base: u32,
}

impl Iterator for OneBits {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.word == 0 {
            return None;
        }
        let tz = self.word.trailing_zeros();
        self.word &= self.word - 1;
        Some(self.base + tz)
    }
}
