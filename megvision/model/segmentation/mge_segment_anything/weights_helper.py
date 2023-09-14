import os

import numpy as np

import megengine as mge


def _set_model_from_mge_weights(model, mge_checkpoint):
    with open(mge_checkpoint, "rb") as f:
        mge_state_dict = mge.load(f)
    model.load_state_dict(mge_state_dict, strict=False)
    return model


def _set_model_from_torch_weights(model, torch_checkpoint):
    import torch

    mge_state_dict = model.state_dict()

    with open(torch_checkpoint, "rb") as f:
        torch_state_dict = torch.load(f)

    either_have = set(mge_state_dict.keys()) | set(torch_state_dict.keys())
    both_have = set(mge_state_dict.keys()) & set(torch_state_dict.keys())
    only_mge_have = set(mge_state_dict.keys()) - set(torch_state_dict.keys())
    only_torch_have = set(torch_state_dict.keys()) - set(mge_state_dict.keys())

    if not either_have == both_have:
        if len(only_mge_have) != 0:
            pass
            # print(f"mge have but torch miss:")
            # for k in only_mge_have:
            #     print(f"    {k}: {mge_state_dict[k].shape}")
        if len(only_torch_have) != 0:
            pass
            # print(f"torch have but mge miss:")
            # for k in only_torch_have:
            #     print(f"    {k}: {torch_state_dict[k].shape}")

    processed_state_dict = {}
    for k in both_have:
        mv = mge_state_dict[k]
        tv = torch_state_dict[k]
        if mv.shape != tuple(tv.shape):
            if np.prod(mv.shape) != np.prod(tv.shape):
                print(f"{k}: mge-shape{mv.shape}, torch-shape{tuple(tv.shape)}")
                continue
            tv = tv.reshape(mv.shape)
        processed_state_dict[k] = np.array(tv.cpu().numpy(), dtype=mv.dtype)
    if not os.path.exists(torch_checkpoint.replace(".pth", ".pkl")):
        with open(torch_checkpoint.replace(".pth", ".pkl"), "wb") as f:
            mge.save(processed_state_dict, f)
    model.load_state_dict(processed_state_dict, strict=False)
    return model


def set_model_weights(model, model_checkpoint):
    if model_checkpoint.endswith("pkl"):
        return _set_model_from_mge_weights(model, model_checkpoint)
    else:
        assert model_checkpoint.endswith("pth")
        return _set_model_from_torch_weights(model, model_checkpoint)
