pub mod crop;
pub mod pad;
pub mod resize;

pub use crop::*;
pub use pad::*;
pub use resize::*;

pub trait Transform<T> {
    fn apply(&self, target: &T) -> T;

    fn apply_iter(&self, target: impl Iterator<Item = T>) -> impl Iterator<Item = T> {
        target.map(|t| self.apply(&t))
    }
}
