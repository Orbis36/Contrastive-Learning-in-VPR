import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

def build_optimizer(model, opt_cfg):
    if opt_cfg.NAME == 'ADAM':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_cfg.LR)
    elif opt_cfg.NAME == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt_cfg.LR,
                momentum=opt_cfg.MOMENTUM, weight_decay=opt_cfg.WEIGHTDECAY)
    else:
        ValueError('Unknown optimizer: ' + opt_cfg.NAME)

    return optimizer
    

def build_scheduler(optimizer, opt_cfg):
    if opt_cfg.SHEDULER_METHOD == 'EQUALGAP':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt_cfg.LRSTEP, gamma=opt_cfg.LRGAMMA)
    else:
        ValueError('Unknown scheduler: ' + opt_cfg.SHEDULER_METHOD)
    return scheduler
