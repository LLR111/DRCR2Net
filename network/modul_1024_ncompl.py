
from backbone import resnext
from res import res as res
from att import *


class NET(nn.Module):
    def __init__(self,train=True,out_c=64):
        super(NET, self).__init__()
        self.training = train
        self.net = resnext.resnext_101_32x4d
        c1,c2,c3,c4,c5 = 1024,512,128,128,128
        self.at5 = CA(c1)
        self.at4 = CA(c2)
        self.at3 = CA(c3)
        self.at2 = CA(c4)
        self.at1 = CA(c5)
        self.crl5 = res(c1, out_c)
        self.crl4 = res(c2, out_c)
        self.crl3 = res(c3, out_c)
        self.crl2 = res(c4, out_c)
        self.crl1 = res(c5, out_c)
    def forward(self, x):
        f1,y = self.net(x)
        f5_c, f5_s = self.at5(y[0])
        f4_c, f4_s = self.at4(y[1])
        f3_c, f3_s = self.at3(y[2])
        f2_c, f2_s = self.at2(y[3])
        f1_c, f1_s = self.at1(f1)
        f5_c_m = self.crl5(f5_c)
        f5_s_m = self.crl5(f5_s)
        f4_c_m = self.crl4(f4_c+f5_c_m) + f5_c_m
        f4_s_m = self.crl4(f4_s+f5_s_m) + f5_s_m
        f3_c_m = self.crl3(f3_c+f4_c_m) + f4_c_m
        f3_s_m = self.crl3(f3_s+f4_s_m) + f4_s_m
        f2_c_m = self.crl2(f2_c+f3_c_m) + f3_c_m
        f2_s_m = self.crl2(f2_s+f3_c_m) + f3_s_m
        f1_c_m = self.crl1(f1_c+f2_c_m) + f2_c_m
        f1_s_m = self.crl1(f1_s+f2_c_m) + f2_s_m
        final_map = (f1_s_m + f1_c_m)/2
        return final_map










