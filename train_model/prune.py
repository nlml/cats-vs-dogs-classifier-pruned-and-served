import torch


def prune_conv_layer(conv, keep=None, p_keep=1.0):
    # General function to return a pruned version of the input conv layer,
    # plus the indices of the weights that were pruned.
    sd = conv.state_dict()
    if keep is not None:
        sd['weight'] = sd['weight'][:, keep]
    if p_keep < 1.0:
        num_to_keep = int(p_keep * sd['weight'].shape[0])
        keep = torch.abs(sd['weight']).sum(3).sum(2).sum(1).argsort()[-num_to_keep:]
        sd['weight'] = sd['weight'][keep]
        if 'bias' in sd.keys():
            sd['bias'] = sd['bias'][keep]
    out_shape, in_shape = sd['weight'].shape[:2]
    out = conv.__class__(
        in_shape,
        out_shape,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding=conv.padding,
    )
    out.load_state_dict(sd)
    return out, keep


def prune_fire_layer(fire, keep, p_keep, prune_output_shape=True):
    # This prunes (1 - p_keep) of the output weights from the expand1x1
    # and expand3x3 layers of a 'Fire' layer in the SqueezeNet architecture.
    # It also prunes the input weights if the previous layer's output was pruned.
    expand1x1_planes = fire.expand1x1.weight.shape[0]
    fire.squeeze, _ = prune_conv_layer(fire.squeeze, keep, 1.0)
    if prune_output_shape:
        fire.expand1x1, keep0 = prune_conv_layer(fire.expand1x1, None, p_keep)
        fire.expand3x3, keep1 = prune_conv_layer(fire.expand3x3, None, p_keep)
        keep = torch.cat([keep0, keep1 + expand1x1_planes])
    else:
        keep = None
    return keep


def prune_squeezenet(model, p_keep):
    # Prunes (1 - p_keep) of weights from a SqueezeNet model.
    keep = None
    num_fire, count = 0, 0
    for fire in model.features:
        if str(fire.__class__).endswith("Fire'>"):
            num_fire += 1
    for fire in model.features:
        if str(fire.__class__).endswith("Fire'>"):
            count += 1
            keep = prune_fire_layer(fire, keep, p_keep, count != num_fire)
    return model
