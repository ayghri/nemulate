import torch


def polar_reflection(
    input_tensor,
    north_padding,
    south_padding,
    lat_dim=-2,
    long_dim=-1,
):
    """
    Map in the format
         . North
        / \\
    1 2 3 4 5 6  |
    7 8 9 0 1 2  |  = Map 
    3 4 5 6 7 8  |
       \\ /
         . South

    with north_padding = 2, south_padding=1
    first reflect the north and south pads
    7 8 9 0 1 2
    1 2 3 4 5 6
    ... Map
    3 4 5 6 7 8

    Then shift them by num_long // 2, so that the final result is:
    
    0 1 2 7 8 9
    4 5 6 1 2 3
    ... Map
    6 7 8 3 4 5 

    """
    assert north_padding > 0 or south_padding > 0, (
        "At least one of north_padding or south_padding must be greater than 0."
    )

    shift = input_tensor.shape[long_dim] // 2
    cat_tensors = []

    if north_padding > 0:
        # 1. Slice the source from the top side of the input
        # start from 0, not the same as pytorch pad reflect that starts from 1
        n_pad = input_tensor.narrow(lat_dim, 0, north_padding)
        # 2. Flip it vertically to create the reflection
        n_pad = torch.flip(n_pad, dims=[lat_dim])
        # 3. Roll the reflected slice horizontally
        cat_tensors.append(torch.roll(n_pad, shifts=shift, dims=[long_dim]))

    cat_tensors.append(input_tensor)

    if south_padding > 0:
        s_pad = input_tensor.narrow(lat_dim, -south_padding, south_padding)
        s_pad = torch.flip(s_pad, dims=[lat_dim])
        s_pad = torch.roll(s_pad, shifts=shift, dims=[long_dim])
        cat_tensors.append(s_pad)

    return torch.cat(cat_tensors, dim=lat_dim)


# @torch.compile
def translated_reflection(
    input_tensor: torch.Tensor,
    reflect_dim: int,
    padding: tuple[int, int],
    shift_dim: int = -1,
    shift: int = 1,
) -> torch.Tensor:
    """
    Pads dimension dim of a tensor with shift-translated reflection padding.

    Args:
        input_tensor (Tensor): The input tensor.
        dim: int: The dimension to apply the padding on.
        pad (tuple[int, int]): A 2-element tuple of (pad_left, pad_right).
        v_shift (int): The number of positions to shift vertically.

    Returns:
        Tensor: The padded tensor.
    """
    padding_l, padding_r = padding
    assert reflect_dim < input_tensor.dim(), (
        "reflect_dim must be less than ndim of the tensor."
    )
    assert shift_dim < input_tensor.dim(), (
        "shift_dim must be less than ndim of the tensor."
    )

    pieces_to_cat = []

    # --- Left Padding ---
    if padding_l > 0:
        # 1. Slice the source from the left side of the input
        l_pad_source = input_tensor.narrow(reflect_dim, 0, padding_l)
        # 2. Flip it horizontally to create the reflection
        l_pad_reflected = torch.flip(l_pad_source, dims=[reflect_dim])
        # 3. Roll the reflected slice vertically
        l_pad = torch.roll(l_pad_reflected, shifts=shift, dims=[shift_dim])
        pieces_to_cat.append(l_pad)

    pieces_to_cat.append(input_tensor)

    # --- Right Padding ---
    if padding_r > 0:
        r_pad_source = input_tensor.narrow(reflect_dim, -padding_r, padding_r)
        r_pad_reflected = torch.flip(r_pad_source, dims=[reflect_dim])
        r_pad = torch.roll(r_pad_reflected, shifts=shift, dims=[shift_dim])
        pieces_to_cat.append(r_pad)

    return torch.cat(pieces_to_cat, dim=reflect_dim)
