//! The five ITG difficulty slots.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Difficulty {
    Beginner,
    Easy,
    Medium,
    Hard,
    Challenge,
}

impl Difficulty {
    pub const ALL: [Difficulty; 5] = [
        Difficulty::Beginner,
        Difficulty::Easy,
        Difficulty::Medium,
        Difficulty::Hard,
        Difficulty::Challenge,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    /// Name as written in simfiles.
    pub fn name(self) -> &'static str {
        match self {
            Difficulty::Beginner => "Beginner",
            Difficulty::Easy => "Easy",
            Difficulty::Medium => "Medium",
            Difficulty::Hard => "Hard",
            Difficulty::Challenge => "Challenge",
        }
    }

    /// Parses user input or simfile names ("expert" is the DDR name for Challenge).
    pub fn parse(s: &str) -> Result<Difficulty> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "beginner" | "b" | "novice" => Difficulty::Beginner,
            "easy" | "e" | "basic" | "light" => Difficulty::Easy,
            "medium" | "m" | "standard" | "another" | "trick" => Difficulty::Medium,
            "hard" | "h" | "heavy" | "maniac" => Difficulty::Hard,
            "challenge" | "c" | "x" | "expert" | "smaniac" | "oni" => Difficulty::Challenge,
            _ => bail!("unknown difficulty {s:?} (beginner, easy, medium, hard, challenge)"),
        })
    }

    /// Constant mixed into the user seed so each difficulty has its own random stream.
    pub fn seed_salt(self) -> u64 {
        (self.index() as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

/// Parses a comma-separated list; "all" selects every difficulty. Sorted, deduplicated.
pub fn parse_list(s: &str) -> Result<Vec<Difficulty>> {
    let mut out = Vec::new();
    for part in s.split(',').filter(|p| !p.trim().is_empty()) {
        if part.trim().eq_ignore_ascii_case("all") {
            out.extend(Difficulty::ALL);
        } else {
            out.push(Difficulty::parse(part)?);
        }
    }
    out.sort();
    out.dedup();
    if out.is_empty() {
        bail!("no difficulty selected");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lists() {
        assert_eq!(
            parse_list("hard, easy,E").unwrap(),
            vec![Difficulty::Easy, Difficulty::Hard]
        );
        assert_eq!(parse_list("all").unwrap().len(), 5);
        assert!(parse_list("impossible").is_err());
        assert_eq!(Difficulty::parse("Expert").unwrap(), Difficulty::Challenge);
    }
}
