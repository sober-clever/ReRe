def get_sentence_start_positions(input_ids, tokenizer):
    cut_indexes = []

    for token_id in input_ids:
        # Find indices of eos_token_id in the current batch element
        # + 1 so to get the index of the beginning of a sentence
        batch_cut_indexes = ((token_id == tokenizer.eos_token_id).nonzero(as_tuple=True)[0] + 1).tolist()

        if tokenizer.padding_side == "right":
            final_padding_idx = -1
        else:
            _flatten_pad_indexes = (token_id == tokenizer.pad_token_id).nonzero().flatten()

            if len(_flatten_pad_indexes) == 0:
                final_padding_idx = -1
            else:
                final_padding_idx = _flatten_pad_indexes[:batch_cut_indexes[0]][-1].item()

        batch_cut_indexes = [final_padding_idx + 1] + batch_cut_indexes

        cut_indexes.append(batch_cut_indexes)
    return cut_indexes


def process_metas(feature_meta_, eos_token, meta_cutoff_len):
    _all_meta_ids = []
    for i, meta_str_ in enumerate(feature_meta_):
        if meta_cutoff_len is not None:
            meta_str_ = meta_str_[:meta_cutoff_len]
        _all_meta_ids.append(meta_str_ + eos_token)
    return _all_meta_ids