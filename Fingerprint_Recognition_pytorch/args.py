import os

class data_config:
    # data_path = '/disks/disk2/lishengyan/MyProject/Fingerprint_Recognition_pytorch/dataset_FVC2000_DB4_B/dataset/'
    data_path ='/disks/disk2/lishengyan/dataset/fingerprint/'
    model_name = "mobilenet_baseline"
    MODEL_PATH = '/disks/disk2/lishengyan/MyProject/Fingerprint_Recognition_pytorch/ckpts/FVC2000/resnet50/mobilenet_baseline/'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    WORKERS = 5
    epochs = 100#resnet:160,wresnet:200,resnext:300
    num_class = 100
    input_size = 128
    batch_size = 32
    delta =0.00001
    rand_seed=40
    lr=0.1
    warm=1#warm up training phase
    optimizer = "torch.optim.SGD"
    optimizer_parm = {'lr': lr,'momentum':0.9, 'weight_decay':5e-4, 'nesterov':False}
    # optimizer = "torch.optim.AdamW"
    # optimizer_parm = {'lr': 0.001, 'weight_decay': 0.00001}
    #scheduler ="torch.optim.lr_scheduler.MultiStepLR"
    #scheduler_parm ={'milestones':[60,120,160], 'gamma':0.2}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max': 100, 'eta_min': 1e-4}
    loss_f ='torch.nn.CrossEntropyLoss'
    loss_dv = 'torch.nn.KLDivLoss'
    loss_fn = 'torch.nn.BCELoss'
