pub mod numbers;
#[macro_use]
pub mod array;

#[cfg(test)]
mod tests {
    #[test]
    fn test_backward() {
        assert_eq!(2 + 2, 4);
    }
}
