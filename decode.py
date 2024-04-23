import time  # 导入时间模块
import numpy as np  # 导入NumPy库，这是Python常用的科学计算库
import torch  # 导入PyTorch库，这是一个开源的深度学习平台
import argparse  # 导入argparse库，这个库是用来管理命令行参数输入的
from pathlib import Path  # 从pathlib库中导入Path，这个库是用来处理文件路径的
from hydra import initialize, compose  # 从hydra库中导入initialize和compose，这个库是用来帮助你更好地组织你的程序配置的
from tqdm import tqdm  # 从tqdm库中导入tqdm，这个库是用来在Python长循环中添加进度提示信息的，让你的程序看起来更酷炫
from collections import deque  # 从collections库中导入deque，这是一个双端队列

from data_preproc.Octree import DeOctree, dec2bin  # 从data_preproc.Octree模块中导入DeOctree和dec2bin，这是你自定义的模块，用于处理八叉树数据
from data_preproc import pt  # 从data_preproc模块中导入pt，这是你自定义的模块，用于处理点云数据
from models import OctAttention  # 从models模块中导入OctAttention，这是你自定义的模块，用于处理模型的注意力机制
import numpyAc  # 导入numpyAc，这是你自定义的模块，用于处理numpy数组的操作

def extract_max_level(file):  # 定义一个函数，用于提取文件名中的最大层级信息
    if file[-5] == "9":  # 如果文件名的倒数第五个字符是"9"
        return 9  # 那么返回9
    return int(file[-6:-4])  # 否则，返回文件名的倒数第六个到倒数第四个字符转换成的整数

def cal_pos(parent_pos, i, cur_level, max_level):  # 定义一个函数，用于计算位置
    pos = torch.zeros_like(parent_pos)  # 创建一个与parent_pos形状相同，但是全为0的张量
    parent_pos = parent_pos * (2 ** max_level)  # 将parent_pos乘以2的max_level次方
    parent_pos = torch.round(parent_pos).long()  # 对parent_pos进行四舍五入，并转换为长整型
    xyz = dec2bin(i, count=3)  # 将i转换为二进制表示，得到xyz
    unit = 2 ** (max_level - cur_level + 1)  # 计算单位长度
    for i in range(3):  # 对于xyz的每一个维度
        pos[i] = (xyz[i] * unit + parent_pos[i]) / (2 ** max_level)  # 计算pos的对应维度的值
    return pos  # 返回pos

def decodeOct(binfile, oct_data_seq, model, context_size, level_k):  # 定义一个函数，用于解码八叉树数据
    """
    description: decode bin file to occupancy code
    param {str;input bin file name} binfile
    param {N*1 array; occupancy code, only used for check} oct_data_seq
    param {model} model
    param {int; Context window length}context_size
    return {N*1,float}occupancy code,time
    """
    model.eval()  # 将模型设置为评估模式
    oct_data_seq -= 1  # 将oct_data_seq减1
    max_level = extract_max_level(binfile)  # 提取最大层级信息
    cur_level = 1  # 设置当前层级为1

    with torch.no_grad():  # 关闭梯度计算
        elapsed = time.time()  # 记录当前时间

        nodeQ = deque()  # 创建一个双端队列nodeQ
        posQ = deque()  # 创建一个双端队列posQ
        oct_seq = []  # 创建一个空列表oct_seq
        oct_len = len(oct_data_seq)  # 计算oct_data_seq的长度

        ipt = torch.zeros((context_size, level_k, 3)).long().cuda()  # 创建一个全为0的张量ipt，并将其转移到GPU上
        ipt[:, :, 0] = 255  # 将ipt的第一维度全设为255
        ipt[-1, -1, 1:3] = 1  # 将ipt的最后一个元素的后两个维度设为1
        ipt_pos = torch.zeros((context_size, level_k, 3)).cuda()  # 创建一个全为0的张量ipt_pos，并将其转移到GPU上

        output = model(ipt[True], ipt_pos[True])  # 将ipt和ipt_pos输入模型，得到输出
        freqsinit = torch.softmax(output[0, -1], 0).cpu().numpy()  # 对输出进行softmax操作，并将其转移到CPU上，再转换为numpy数组

        dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)  # 进行算术解码

        root = decodeNode(freqsinit, dec)  # 解码节点
        node_id = 0  # 设置节点id为0

        ipt[-1, -1, 0] = root  # 将ipt的最后一个元素的第一个维度设为root
        nodeQ.append(ipt[-1, -(level_k - 1):].clone())  # 将ipt的最后一个元素的后(level_k - 1)个元素添加到nodeQ中
        posQ.append(ipt_pos[-1, -(level_k - 1):].clone())  # 将ipt_pos的最后一个元素的后(level_k - 1)个元素添加到posQ中
        oct_seq.append(root)  # 将root添加到oct_seq中

        with tqdm(total=oct_len) as pbar:  # 创建一个进度条
            while True:  # 不断循环
                ancients = nodeQ.popleft()  # 从nodeQ中弹出一个元素，赋值给ancients
                ancient_pos = posQ.popleft()  # 从posQ中弹出一个元素，赋值给ancient_pos
                parent_pos = ancient_pos[-1]  # 取ancient_pos的最后一个元素，赋值给parent_pos

                childOcu = dec2bin(ancients[-1, 0] + 1)  # 将ancients的最后一个元素的第一个维度加1，转换为二进制表示，得到childOcu
                childOcu.reverse()  # 将childOcu反转
                cur_level = ancients[-1][1] + 1  # 将ancients的最后一个元素的第二个维度加1，赋值给cur_level
                # TODO level变了要清空
                for i in range(8):  # 对于childOcu的每一个元素
                    if childOcu[i]:  # 如果该元素为真
                        cur_feat = torch.vstack((ancients, torch.Tensor([[255, cur_level, i + 1]]).cuda()))  # 将ancients和一个新的张量垂直堆叠起来，得到cur_feat
                        cur_pos = cal_pos(parent_pos, i, cur_level, max_level)  # 计算当前位置，得到cur_pos
                        cur_pos = torch.vstack((ancient_pos.clone(), cur_pos))  # 将ancient_pos和cur_pos垂直堆叠起来，得到新的cur_pos

                        # shift context_size window
                        ipt[:-1] = ipt[1:].clone()  # 将ipt的前一部分设为ipt的后一部分
                        ipt[-1] = cur_feat  # 将ipt的最后一个元素设为cur_feat
                        ipt_pos[:-1] = ipt_pos[1:].clone()  # 将ipt_pos的前一部分设为ipt_pos的后一部分
                        ipt_pos[-1] = cur_pos  # 将ipt_pos的最后一个元素设为cur_pos

                        output = model(ipt[True], ipt_pos[True])
                        probabilities = torch.softmax(output[0, -1], 0).cpu().numpy()
                        root = decodeNode(probabilities, dec)

                        node_id += 1
                        pbar.update(1)

                        ipt[-1, -1, 0] = root
                        nodeQ.append(ipt[-1, 1:].clone())
                        posQ.append(ipt_pos[-1, 1:].clone())
                        if node_id == oct_len:
                            return oct_seq, time.time() - elapsed
                        oct_seq.append(root)
                        assert oct_data_seq[node_id] == root  # for check


def decodeNode(pro, dec):  # 定义一个函数，用于解码节点
    root = dec.decode(np.expand_dims(pro, 0))  # 对pro进行扩展维度后进行解码，得到root
    return root  # 返回root

def main(args):  # 定义主函数
    root_path = args.ckpt_path.split("ckpt")[0]  # 获取根路径
    test_output_path = (
        root_path + "test_output" + args.ckpt_path.split("ckpt")[1][:-1] + "/"
    )  # 获取测试输出路径
    cfg_path = Path(root_path, ".hydra")  # 获取配置文件路径
    initialize(config_path=str(cfg_path))  # 初始化配置
    cfg = compose(config_name="config.yaml")  # 组合配置

    model = OctAttention.load_from_checkpoint(args.ckpt_path, cfg=cfg).cuda()  # 加载模型
    for ori_file in args.test_files:  # 对于每一个测试文件
        ori_file = Path(ori_file)  # 获取文件路径
        binfile = test_output_path + ori_file.stem + ".bin"  # 获取二进制文件路径

        # load ori data
        npy_path = str(ori_file).rsplit(".")[0]  # 获取npy文件路径
        oct_data_seq = np.load(npy_path + ".npy")[:, -1:, 0]  # 加载npy文件

        code, elapsed = decodeOct(
            binfile, oct_data_seq, model, cfg.model.context_size, cfg.model.level_k
        )  # 解码八叉树数据
        print("decode succee,time:", elapsed)  # 打印解码成功和所用时间
        print("oct len:", len(code))  # 打印八叉树长度

        # DeOctree
        pt_rec = DeOctree(code)  # 对code进行反八叉树操作，得到pt_rec
        # Dequantization
        # TODO fit for lidar dataset
        offset = 0  # 设置偏移量为0
        qs = 1  # 设置量化步长为1
        pt_rec = pt_rec * qs + offset  # 对pt_rec进行反量化操作
        pt.write_ply_data(test_output_path + ori_file.stem + ".ply", pt_rec)  # 将pt_rec写入ply文件

def get_args():  # 定义一个函数，用于获取命令行参数
    parser = argparse.ArgumentParser()  # 创建一个命令行解析器
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="example: outputs/obj/2023-04-28/10-43-45/ckpt/epoch=7-step=64088.ckpt",
    )  # 添加一个命令行参数
    parser.add_argument(
        "--test_files",
        nargs="*",
        default=["data/obj/mpeg/8iVLSF_910bit/boxer_viewdep_vox9.ply"],
    )  # 添加一个命令行参数
    parser.add_argument("--sequential_enc", action="store_true")  # 添加一个命令行参数
    parser.add_argument("--level_wise", action="store_true")  # 添加一个命令行参数
    return parser.parse_args()  # 解析命令行参数并返回

if __name__ == "__main__":  # 如果当前脚本是主程序
    args = get_args()  # 获取命令行参数
    main(args)  # 调用主函数

