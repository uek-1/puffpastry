use proc_macro::{Delimiter, TokenStream, TokenTree};

pub fn shove_all_array(input: &TokenStream) -> Vec<f32> {
    let mut data = vec![];
    input.clone().into_iter().for_each(|x| match x {
        TokenTree::Punct(_) => (),
        TokenTree::Ident(_) => (),
        TokenTree::Literal(x) => data.push(x.to_string().parse().unwrap()),
        TokenTree::Group(group) => data.append(&mut shove_all_array(&group.stream())),
    });

    data
}

pub fn get_shape(input: &TokenStream) -> Vec<usize> {
    let mut shape: Vec<usize> = vec![];
    let mut inner = None;

    input.clone().into_iter().for_each(|x| match x {
        TokenTree::Punct(_) => (),
        TokenTree::Ident(_) => (),
        TokenTree::Literal(_) => match shape.get_mut(0) {
            Some(x) => *x += 1,
            None => shape.push(1),
        },
        TokenTree::Group(group) if group.delimiter() == Delimiter::Bracket => {
            if inner.is_none() {
                inner = Some(group.stream());
            }

            match shape.get_mut(0) {
                Some(x) => *x += 1,
                None => shape.push(1),
            }
        }
        TokenTree::Group(_) => (),
    });

    if let Some(stream) = inner {
        shape.extend(get_shape(&stream).into_iter());
    }

    shape
}
