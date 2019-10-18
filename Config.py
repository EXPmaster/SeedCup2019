import torch
import time
import os


class Config(object):
    def __init__(self):
        self.USE_CUDA = torch.cuda.is_available()
        self.NUM_EPOCHS = 100

        self.TRAIN_BATCH_SIZE = 32
        self.VAL_BATCH_SIZE = 32
        self.TEST_BATCH_SIZE = 64

        self.LR = 1e-3  # default learning rate

        self.EMBEDDING_DIM = 350 # 686
        self.LINER_HID_SIZE = 1024
        self.INPUT_SIZE = 13

        self.OUTPUT_DIM = 1

        self.LOSS_1_WEIGHT = 1

        self.plat_form_range = 5
        self.biz_type_range = 6
        self.product_id_range = 26978
        self.cate1_id_range = 27
        self.cate2_id_range = 272
        self.cate3_id_range = 1609
        self.seller_uid_range = 1000
        self.company_name_range = 276807
        self.lgst_company_range = 17
        self.warehouse_id_range = 12
        self.shipped_prov_id_range = 28
        self.shipped_city_id_range = 118
        self.rvcr_prov_name_range = 33
        self.rvcr_city_name_range = 435


