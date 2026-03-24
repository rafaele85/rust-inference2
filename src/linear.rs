use crate::common_types::Arrf32;

pub fn add_to_vector(input: &mut Arrf32, vector_to_add: &Arrf32) -> () {
    for i in 0..input.len() {
        input[i] += vector_to_add[i];
    }
}
