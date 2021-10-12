
def get_fusion_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet

    net = FusionNet(num_class=num_class)
    return net