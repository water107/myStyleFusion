from PIL import Image
from Metric import *
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def evaluation_one(ir_name, vi_name, f_name):
    f_img = Image.open(f_name).convert('L')
    ir_img = Image.open(ir_name).convert('L')
    vi_img = Image.open(vi_name).convert('L')
    f_img_int = np.array(f_img).astype(np.int32)

    f_img_double = np.array(f_img).astype(np.float32)
    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)

    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
    MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
    PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)

    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    SF = SF_function(f_img_double)

    CC = CC_function(ir_img_double, vi_img_double, f_img_double)
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
    # Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    # Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)
    # SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
    # MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)
    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR

if __name__ == '__main__':
    f_name = r"D:\Doc\期刊\航天控制\2\epoch500.jpg"
    ir_name = r'data/s2.jpg'
    vi_name = r'data/t2.jpg'
    EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR = evaluation_one(ir_name, vi_name, f_name)
    print('信息熵EN:', round(EN, 4))
    # print('互信息:', round(MI, 4))
    # print('均方误差MSE:', round(MSE, 4))
    print('峰值信噪比PSNR:', round(PSNR, 4))

    print('标准差SD:', round(SD, 4))
    print('平均梯度AG:', round(AG, 4))
    print('空间频率SF:', round(SF, 4))

    # print('相关系数CC:', round(CC, 4))
    # print('差异相关和SCD:', round(SCD, 4))
    # print('视觉保证度VIF:', round(VIF, 4))

    # print('Qabf:', round(Qabf, 4))
    # print('Nabf:', round(Nabf, 4))
    # print('SSIM:', round(SSIM, 4))
    # print('MS_SSIM:', round(MS_SSIM, 4))
