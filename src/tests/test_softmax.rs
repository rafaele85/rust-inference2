pub fn test_softmax() {
    let result = softmax(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let formatted: Vec<String> = result.iter().map(|v| format!("{:.4}", v)).collect();
    println!("{}", formatted.join(" "));
}