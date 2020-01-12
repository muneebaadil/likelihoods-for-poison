from importlib import import_module
import torch


def get_model(opts, logger):
    if opts.model == 'alexnet':
        from torchvision import models
        model = models.alexnet(pretrained=False)
    elif opts.model == 'lenet5':
        import model.lenet5
        model = model.lenet5.LeNet5(opts.n_feats, opts.n_classes)
    elif opts.model == 'cnn':
        import model.cnn as cnn
        model = cnn.CNNModel(opts.n_feats)
    elif opts.model == 'net':
        import model.net as net
        model = net.MNISTNet()
    else:
        raise NotImplementedError()

    if opts.ckpt_path:
        model.load_state_dict(torch.load(opts.ckpt_path), strict=True)
        logger.logger.info("Weights loaded from %s" % opts.ckpt_path)

    if opts.use_cuda:
        model = model.cuda()
        if opts.n_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=opts.gpu_ids)

    if opts.print_model:
        print(model)

    return model
