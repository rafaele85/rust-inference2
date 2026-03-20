use crate::transformer::{transformer, TransformerArgs};
pub fn test_single_token(args: TransformerArgs)  {
    let top_id = transformer(args);

    println!("{}", top_id);
}