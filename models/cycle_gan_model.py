import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import lpips
class CycleGANModel(BaseModel):
    """
    这个类实现了CycleGAN模型，用于在没有成对数据的情况下学习图像到图像的转换。

    模型训练需要 '--dataset_mode unaligned' 数据集。
    默认情况下，它使用 '--netG resnet_9blocks' ResNet生成器，
    一个 '--netD basic' 判别器（pix2pix中引入的PatchGAN），
    和一个最小平方GAN目标（'--gan_mode lsgan'）。

    CycleGAN论文：https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的数据集特定选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否是训练阶段。你可以使用这个标志来添加训练特定或测试特定的选项。

        返回:
            修改后的解析器。

        对于CycleGAN，除了GAN损失外，我们还引入了lambda_A, lambda_B和lambda_identity用于以下损失。
        A（源域），B（目标域）。
        生成器：G_A: A -> B; G_B: B -> A。
        判别器：D_A: G_A(A) vs. B; D_B: G_B(B) vs. A。
        前向循环损失： lambda_A * ||G_B(G_A(A)) - A|| （论文中的公式(2)）
        后向循环损失： lambda_B * ||G_A(G_B(B)) - B|| （论文中的公式(2)）
        身份损失（可选）： lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) （论文第5.2节“从绘画生成照片”）
        原始CycleGAN论文中没有使用Dropout。
        """
        parser.set_defaults(no_dropout=True)  # 默认CycleGAN不使用dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='循环损失的权重 (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='循环损失的权重 (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='使用身份映射。设置lambda_identity不为0会影响身份映射损失的权重。例如，如果身份损失的权重应比重建损失小10倍，请设置lambda_identity = 0.1')

        return parser

#############################
# D_A：判别器A的损失。判别器A用于区分真实的目标域图像和生成的目标域图像。

# G_A：生成器A的损失。生成器A用于将源域图像转换为目标域图像。

# cycle_A：前向循环一致性损失。用于确保源域图像经过生成器A和生成器B的转换后能够重建回原始的源域图像。

# idt_A：生成器A的身份损失。用于确保当目标域图像输入到生成器A时，生成器A能够输出与输入相同的图像。这在某些情况下可以帮助保持图像的颜色和其他特征。

# D_B：判别器B的损失。判别器B用于区分真实的源域图像和生成的源域图像。

# G_B：生成器B的损失。生成器B用于将目标域图像转换为源域图像。

# cycle_B：后向循环一致性损失。用于确保目标域图像经过生成器B和生成器A的转换后能够重建回原始的目标域图像。

# idt_B：生成器B的身份损失。用于确保当源域图像输入到生成器B时，生成器B能够输出与输入相同的图像。这在某些情况下可以帮助保持图像的颜色和其他特征。

######################################
    def __init__(self, opt):
        """初始化CycleGAN类。

        参数:
            opt (Option class)-- 存储所有实验标志；需要是BaseOptions的子类
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # 如果使用身份损失，我们还会可视化idt_B=G_A(B)和idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # 合并A和B的可视化
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_networks>和<BaseModel.load_networks>。
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # 在测试时，只加载生成器
            self.model_names = ['G_A', 'G_B']

        # 定义网络（生成器和判别器）
        # 命名与论文中不同。
        # 代码（vs. 论文）：G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # 定义判别器
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # 仅当输入和输出图像具有相同数量的通道时才有效
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储先前生成的图像
            self.fake_B_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区以存储先前生成的图像
            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # 定义GAN损失。
            self.criterionCycle = lpips.LPIPS(net='alex').to(self.device)  # 定义循环一致性损失
            self.criterionIdt = torch.nn.L1Loss()
            # 初始化优化器；调度器将由<BaseModel.setup>函数自动创建。
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。

        参数:
            input (dict): 包含数据本身及其元数据信息。

        选项 'direction' 可用于交换域A和域B。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """运行前向传递；由<optimize_parameters>和<test>函数调用。"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """计算判别器的GAN损失

        参数:
            netD (network)      -- 判别器D
            real (tensor array) -- 真实图像
            fake (tensor array) -- 生成器生成的图像

        返回判别器损失。
        我们还调用loss_D.backward()来计算梯度。
        """
        # 真实图像
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # 生成的图像
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # 合并损失并计算梯度
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """计算判别器D_A的GAN损失"""
        # 从图像池中获取生成的假图像fake_B
        fake_B = self.fake_B_pool.query(self.fake_B)
        # 计算判别器D_A的损失，并将其存储在self.loss_D_A中
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """计算判别器D_B的GAN损失"""
        # 从图像池中获取生成的假图像fake_A
        fake_A = self.fake_A_pool.query(self.fake_A)
        # 计算判别器D_B的损失，并将其存储在self.loss_D_B中
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """计算生成器G_A和G_B的损失"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # 身份损失
        if lambda_idt > 0:
            # 如果输入real_B，G_A应该是身份映射：||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # 如果输入real_A，G_B应该是身份映射：||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN损失 D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN损失 D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # 前向循环损失 || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A).mean() * lambda_A
        # 后向循环损失 || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B).mean() * lambda_B
        # 合并损失并计算梯度
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重；在每次训练迭代中调用"""
        # 前向传递
        self.forward()      # 计算生成的图像和重建图像。
        # G_A和G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # 优化生成器时判别器不需要梯度
        self.optimizer_G.zero_grad()  # 将G_A和G_B的梯度置零
        self.backward_G()             # 计算G_A和G_B的梯度
        self.optimizer_G.step()       # 更新G_A和G_B的权重
        # D_A和D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # 将D_A和D_B的梯度置零
        self.backward_D_A()      # 计算D_A的梯度
        self.backward_D_B()      # 计算D_B的梯度
        self.optimizer_D.step()  # 更新D_A和D_B的权重