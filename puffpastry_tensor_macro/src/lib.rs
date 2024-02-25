mod tensor_macro;
use proc_macro::TokenStream;
use tensor_macro::*;

#[proc_macro]
pub fn tensor(input: TokenStream) -> TokenStream {
    let data = shove_all_array(&input);
    let shape: Vec<usize> = get_shape(&input).into_iter().skip(1).collect();

    let out_str = format!(
        "Tensor {{ data: {:?}.to_vec(), shape: {:?}.to_vec() }}",
        data, shape
    );

    out_str.parse().unwrap()
}
