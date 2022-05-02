#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    # ###################################################################################################################
    # # std import
    import os
    import sys
    # ###################################################################################################################
    # # third party import
    import torch.nn as nn
    # ###################################################################################################################
    # # app specific import
    # import utility_function
    import preprocess
    import init_train_module
    import model_define
    #
    def baseline_func(prepro_func, prepro_params, model_func, model_params, num_cls):
        # set device
        m_device = init_train_module.init_device('gpu', 0)
        ###################################################################################################################
        # set the data set parameters
        m_data_set_path = '.\\pik\\test_22022-03-05-13-36-24_Cell_set_MinMax_pad_labels_formed.pickle'

        m_train_mode = 'train'  # ('pretrain', 'train', 'test', 'finetune')

        #           len(batch_size)
        # pre-train        1
        # other            3
        batch_size = [64, 100, 100]
        m_data_loader_dict = init_train_module.init_data_loader_dict(m_data_set_path, m_train_mode, batch_size)
        ###################################################################################################################
        # set preprocessing
        # preprocess parameter for baseline
        # m_prepro_param = {
        #     'num_classes': 8,
        # }
        # m_prepro = preprocess.BaseProcessing(**m_prepro_param)
        m_prepro = prepro_func(**prepro_params)
        ###################################################################################################################
        # model parameter for baseline
        # m_model = model_define.BaseLine_ResNet
        # m_model_param = {
        #     'in_dim': 1021,
        #     'num_cls': m_prepro_param['num_classes'],
        # }
        # m_init_model = init_train_module.init_model(m_model, m_model_param, m_device)
        m_init_model = init_train_module.init_model(model_func, model_params, m_device)
        ###################################################################################################################
        # optimizer and scheduler set up
        m_optimizer_param = {
            'optimizer_name': 'Adam',
            'lr': 0.0001,
            'betas': (0.9, 0.98),
            'eps': 1e-9,
            'weight_decay': 0,
            'amsgrad': False,
        }
        m_scheduler_param = {
            'scheduler name': 'StepLR',
            'step_size': 16,
            'gamma': 0.95,
            'last_epoch': -1,
            'verbose': False
        }
        m_opt, m_sch = init_train_module.init_optimaizer_scheduler(m_init_model, m_optimizer_param, m_scheduler_param)
        ###################################################################################################################
        # train
        m_log_dir = os.path.join('.\\log', m_train_mode)
        ###################################################################################################################
        # collect hyper parameter
        m_hyper_param = {
            'train_mode': m_train_mode,
            'data_set': m_data_set_path,
            'batch_size': batch_size[0],
            'model_name': m_init_model.model_name,
        }
        ###################################################################################################################
        # set loss function
        BL_Loss_fn = nn.CrossEntropyLoss()
        m_loss_fn_list = [BL_Loss_fn , ]
        ###################################################################################################################
        m_trainer = init_train_module.Model_Run(device=m_device,
                                                train_mode=m_train_mode,
                                                num_epoch=3,
                                                data_loader=m_data_loader_dict,
                                                preprocessor=m_prepro,
                                                model=m_init_model,
                                                loss_fn_list=m_loss_fn_list,
                                                optimizer=m_opt,
                                                scheduler=m_sch,
                                                log_dir=m_log_dir,
                                                hyper_param=m_hyper_param,
                                                num_class=num_cls)

        # train_mode:('pretrain', 'train')
        m_trainer.run()


    m_prepro_param = {'num_classes': 8,}
    m_prepro = preprocess.BaseProcessing

    m_model_func_list = [
        model_define.BaseLine_MCNN,
        # model_define.BaseLine_MLP,
        # model_define.BaseLine_FCN,
        # model_define.BaseLine_ResNet
    ]

    m_model_params = {
        'in_dim': 1021,
        'num_cls': m_prepro_param['num_classes'],
    }

    m_num_cls = 8

    for temp_model_func in m_model_func_list:
        baseline_func(m_prepro, m_prepro_param, temp_model_func, m_model_params, m_num_cls)

    sys.exit(0)
