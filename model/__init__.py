
def get_model(model_name, num_class ,is_first_bn):

    if model_name == 'FaceBagNet':
        from model.FaceBagNet import Net
        net = Net(num_class=num_class, is_first_bn=is_first_bn, type='A')
    elif model_name == 'ConvMixer':
        from model.ConvMixer import ConvMixer as Net
        net = Net(dim = 512, depth = 16, kernel_size=9, patch_size=14, n_classes=num_class)
    elif model_name == 'MLPMixer':
        from model.MLPMixer import MLPMixer as Net
        net = Net(image_size=96, channels=3, patch_size=16, dim=512, depth=16, num_classes=num_class, expansion_factor=4, dropout=0.)
    elif model_name == 'VisionPermutator':
        from model.ViP import Permutator as Net
        net = Net(image_size=96, patch_size=16, dim=512, depth=16, num_classes=num_class, expansion_factor=4, segments=4, dropout=0.)

    return net