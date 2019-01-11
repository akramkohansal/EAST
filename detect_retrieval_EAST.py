#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:31:17 2019

@author: akramkohansal
"""



def detect(weightfile, imgfolder, imgdestination):
    
    model = East()
    model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model = model.cuda()
    init_weights(model, init_type=cfg.init_type)
    cudnn.benchmark = True
    
    criterion = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    
    weightpath = os.path.abspath(cfg.checkpoint)
    
    print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(weightpath))
    checkpoint = torch.load(weightpath)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(weightpath))
    
    
    output_txt_dir_path = predict(model)

    
if __name__ == "__main__":
    
    weightfile = ""
    imgfolder=""
    imgdestination = ""
    detect( weightfile, imgfolder, imgdestination)