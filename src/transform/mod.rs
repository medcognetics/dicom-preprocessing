pub mod crop;
pub mod pad;
pub mod resize;
pub mod volume;

pub use crop::*;
pub use pad::*;
pub use resize::*;
pub use volume::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord(u32, u32);

impl Coord {
    pub fn new(x: u32, y: u32) -> Self {
        Self(x, y)
    }
}

pub trait Transform<T> {
    /// Apply the transform to a target.
    fn apply(&self, target: &T) -> T;

    /// Apply the transform to an iterator of targets.
    fn apply_iter(&self, target: impl Iterator<Item = T>) -> impl Iterator<Item = T> {
        target.map(|t| self.apply(&t))
    }
}

impl From<&Coord> for (u32, u32) {
    fn from(coord: &Coord) -> Self {
        (coord.0, coord.1)
    }
}

impl From<(u32, u32)> for Coord {
    fn from(tuple: (u32, u32)) -> Self {
        Coord::new(tuple.0, tuple.1)
    }
}

pub trait InvertibleTransform<T>: Transform<T> {
    /// Invert the transform.
    fn invert(&self, target: &T) -> T;

    /// Invert the transform over an iterator of targets.
    fn invert_iter(&self, target: impl Iterator<Item = T>) -> impl Iterator<Item = T> {
        target.map(|t| self.invert(&t))
    }
}
