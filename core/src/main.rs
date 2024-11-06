fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_output() {
        // Capture the output of the main function
        let output = std::panic::catch_unwind(|| {
            main();
        });

        // Ensure the main function runs without panicking
        assert!(output.is_ok());
    }
}
