import datetime
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms
from accelerate import Accelerator
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
# from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import LoraConfig, PeftModel,get_peft_model
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import json


IMAGENET_MEAN = [0.5,0.5,0.5]
IMAGENET_STD = [0.5,0.5,0.5]
IMAGE_SIZE = 128 

class CellDataset(Dataset):
    """单细胞数据集"""
    
    def __init__(self, csv_files, image_size=IMAGE_SIZE, filter_cell_type=None, transform=None):
        """_summary_

        Args:
            csv_files (string list): 文件路径名
            image_size (int, optional): Defaults to 512.
            filter_cell_type (_type_, optional): Defaults to None. 如果指定，只保留这类型样本，用于第二阶段的微调
            transform
        """
        # 1. 读取csv文件
        dfs = []
        
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
            
        self.df = pd.concat(dfs,ignore_index=True)
        print(f"原始数据共: {len(self.df)}条")
        
        # 2. 为第二阶段筛选数据集
        if filter_cell_type:
            if isinstance(filter_cell_type, str):
                filter_cell_type = [filter_cell_type]
                
            self.df = self.df[self.df["cell_type"].isin(filter_cell_type)]
            print(f"筛选之后的剩余数据总数为: {len(self.df)}")
            
        # 3. 图像路径是否合理检查（只使用路径存在的图像）
        self.df = self.df[self.df['image_file_path'].apply(lambda x: os.path.exists(x))]
        print(f"图像存在性检查后的有效样本数为:{len(self.df)}")
        
        # 4. 查看数据分布统计
        print("细胞类型统计:")
        print(self.df['cell_type'].value_counts())
        
        self.image_size = image_size
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def pad_image_with_mean(self, image, target_size):
        """使用均值填充将图像调整为目标大小"""
        h, w = image.shape[:2]
        
        if h >= target_size and w >= target_size:
            return cv2.resize(image, (target_size, target_size))
        
        # 每个通道的平均颜色
        mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
        padded_image = np.ones((target_size, target_size, 3), dtype=np.uint8)
        padded_image[:] = mean_color
        
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 等比缩放后的图像
        resized_image = cv2.resize(image, (new_w, new_h))
        
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # 将等比缩放后的图像放在平均颜色的正方形画布上
        padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        
        return padded_image
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        cell_type = row['cell_type']
        image_path = row['image_file_path']
        
        # 读取图像
        original_img = cv2.imread(image_path)
        padded_img = self.pad_image_with_mean(original_img, self.image_size)
        rgb = cv2.cvtColor(padded_img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        
        if self.transform:
            img = self.transform(img)
        caption = self._generate_caption(cell_type=cell_type)
        return {
            "pixel_values":img,
            "caption":caption,
            "cell_type":cell_type
        }
        

        

    def _generate_caption(self, cell_type):
        """提示词制作(针对不同的细胞类型)"""
        prompt = f"a high-quality microscopic image of {cell_type} cell"
        type_ls = ['CD66b', 'CD14', 'WBC', 'CD3']
        
        if cell_type in type_ls:
            prompt = "a high-quality microscopic image of white blood cell, with regular nucleus morphology and smooth cell boundary"
        elif cell_type == 'CTC':
            prompt = f'A high-resolution brightfield microscopy image of a circulating tumour cell {cell_type}, large irregular nucleus, thin cytoplasm rim, irregular cell outline, sharp focus, detailed texture of cell membrane, realistic biomedical microscope photo style.'
        elif cell_type == 'CHC':
            prompt = f'Brightfield micrograph of a cluster of circulating hybrid cells, mixed tumour/immune morphology, {cell_type}, irregular nuclei, abundant cytoplasm granules, realistic biomedical microscope photo style.'
        
        return prompt

def get_transforms(train=True, image_size=IMAGE_SIZE):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

# 验证函数
@torch.no_grad()
def validate(
    unet,
    vae,
    text_encoder,
    tokenizer,
    val_dataloader,
    noise_scheduler,
    accelerator,
):
    unet.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in val_dataloader:
        # 编码图像
        pixels = batch["pixel_values"].to(device=accelerator.device, dtype=torch.float16)
        latents = vae.encode(pixels).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # 添加噪声
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # 随机时间步
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        ).long()
        
        # 添加噪声
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 编码文本
        text_inputs = tokenizer(
            batch['caption'],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(latents.device)
        
        encoder_hidden_states = text_encoder(text_inputs)[0]
        
        # 预测噪声
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # 计算损失
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss/num_batches if num_batches>0 else float('inf')
    return avg_loss

def train_lora(
    train_csv_files,
    val_csv_file,
    # test_csv_file,
    output_dir,
    model_id = 'stabilityai/stable-diffusion-2-base',
    pretrained_lora_path=None,  # 第二阶段用：加载第一阶段的LoRA
    filter_cell_type=None,  # 第二阶段用：只训练癌细胞和CHC
    num_epochs = 100,
    batch_size = 4,
    val_batch_size=8,
    learning_rate=1e-4,
    gradient_accumulation_steps = 4,
    lora_r = 16,
    lora_alpha = 32,
    lora_dropout = 0.1,
    save_every_n_epochs = 20,
    validation_epochs=5, # 每多少个epoch验证一次
    target_modules = ["to_q",
            "to_k", 
            "to_v",
            "to_out.0",],
    early_stopping_patience = 10 # 早停patience
):
    """
    lora 双阶段训练微调。
    第一阶段调用: 
    train_lora(csv_files, output_dir="./stage1_all_cells_lora")
    第二阶段调用:         
    train_lora(csv_files, output_dir="./stage2_chc_lora", 
                   pretrained_lora_path="./stage1_all_cells_lora",
                   filter_cell_type=["CHC"])
                   / filter_cell_type=["CTC"]
    
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, f"tensorboard_logs_{timestamp}")
    
    # =====================================================
    # 多GPU配置 - 使用Accelerate的DistributedDataParallel
    # =====================================================
    
    # 检测GPU数量
    n_gpus = torch.cuda.device_count()
    print("="*70)
    print(f"GPU配置信息")
    print("="*70)
    print(f"可用GPU数量: {n_gpus}")
    if n_gpus > 1:
        print(f"将使用 {n_gpus} 块GPU进行分布式训练（DDP模式）")
        print("注意: 实际batch_size = per_device_batch_size * n_gpus * gradient_accumulation_steps")
        print(f"  per_device_batch_size: {batch_size}")
        print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"  有效全局batch_size: {batch_size * n_gpus * gradient_accumulation_steps}")
    else:
        print("使用单GPU训练")
    print("="*70 + "\n")
    
    # 初始化accelerator,自动做混合精度、梯度累积、多GPU分布式训练
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=log_dir,
    )
    
    # 只在主进程创建TensorBoard writer
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard日志目录: {log_dir}")
        print(f"查看TensorBoard: tensorboard --logdir={log_dir}\n")
    
    # =====================================================
    # lora 配置
    # =====================================================
    
    # 加载Stable Diffusion 2 组件
    if accelerator.is_main_process:
        print("="*50)
        print("加载stable diffusion2 各个组件")
        print("="*50)
    
    # 加载各个组件
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    )
    
    # 冻结原始模型
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 添加lora层
    if accelerator.is_main_process:
        print("\n"+"="*50)
        print("配置PEFT LoRA适配器...")
        print("="*50)

    
    # 使用新的API:PEFT 适配器路线, 为Unet添加LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules, # "to_out.0": 输出投影的第一个线性层
        lora_dropout=lora_dropout,
    )
    
    # 应用lora 到unet
    if pretrained_lora_path:
        # 第二阶段：加载第一阶段的lora 权重
        if accelerator.is_main_process:
            print(f"加载第一阶段的lora权重: {pretrained_lora_path}")
        unet = PeftModel.from_pretrained(unet, pretrained_lora_path,is_trainable=True) #以unet为基座模型，从pretrained_lora_path中加载lora权重
        if accelerator.is_main_process:
            print("LoRA适配器加载成功，开始第二阶段微调")
    else:
        # 第一阶段微调
        unet = get_peft_model(unet, lora_config)
    
    # 打印可训练参数（仅主进程）
    if accelerator.is_main_process:
        unet.print_trainable_parameters()
    
    
    # 以下备注内容为旧API的attention-processor 路线
    # load_attn_procs = {}
    # for name in unet.attn_processors:
    #     # attn1.processor 是自注意力处理器，attn2.processor: 交叉注意力处理器
    #     # 注：在diffuser里，这些注意力层
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")]) # 获得up_blocks. 后面的第一个数字，也就是block_id
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]
        
    #     # load_attn_procs[name] = LoRAAttnProcessor(
    #     #     hidden_size=hidden_size, 
    #     #     cross_attention_dim = cross_attention_dim,
    #     #     rank = rank
    #     # )  
    # unet.set_attn_processor(load_attn_procs)
    
    # 把attention层的processors统一打包成一个可训练的nn.Module, 当传入lora_layers.parameters(), 也不会影响到原大权重
    # lora_layers = AttnProcsLayers(unet.attn_processors)
    
    # print("lora_layers.parameters()")
    # print(lora_layers.parameters())
    # print("-"*50)
    # for i in lora_layers.parameters():
    #     print(i)
    
    
    
    # =====================================================
    # 准备数据集
    # =====================================================
    
    if accelerator.is_main_process:
        print("\n"+"="*50)
        print("准备数据集...")
        print("="*50)
        print("\n[训练集]")
    
    train_dataset = CellDataset(
        csv_files=train_csv_files,
        filter_cell_type=filter_cell_type,
        transform=get_transforms(train=True)
    )
    
    if accelerator.is_main_process:
        print("\n[验证集]")
    
    val_dataset = CellDataset(
        csv_files=val_csv_file,
        filter_cell_type=filter_cell_type,
        transform=get_transforms(train=False),
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # 优化器 (只优化LoRA参数)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr = learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_epochs * len(train_loader),
        eta_min=learning_rate * 0.1
    )
    
    # Accelerate准备 - 这会自动处理多GPU的DDP包装
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    
    # 训练循环
    if accelerator.is_main_process:
        print("\n" + "=" * 50)
        print("开始训练...")
        print("=" * 50)
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
        if n_gpus > 1:
            print(f"每个GPU的实际batch size: {batch_size}")
            print(f"全局有效batch size: {batch_size * n_gpus * gradient_accumulation_steps}")
        print("=" * 50 + "\n")
    
    
    global_step = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0
        
        # tqdm 会根据 len(dataloader) 自动显示已处理 batch / 总 batch
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # 1. 编码图像到latent space
                with torch.no_grad():
                    pixels = batch['pixel_values'].to(device=accelerator.device, dtype=torch.float16)
                    latents = vae.encode(pixels).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    
                # 添加噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # 随机时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device
                ).long()
                
                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents,noise=noise,timesteps=timesteps)
                
                # 编码文本
                text_inputs = tokenizer(
                    batch['caption'],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(latents.device)  # input_ids : 词表索引
                
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_inputs)[0] #给文本索引编码
                    
                # 预测噪声
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 计算损失   .float() 是.to(torch.float32) 的简写，等价于把张量转换为 float32
                loss = F.mse_loss(noise.float(), model_pred.float(),reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step+=1
                
                # 记录到TensorBoard（仅主进程）
                if accelerator.is_main_process and global_step % 10 == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0],global_step)
                
        avg_train_loss = epoch_loss/len(train_loader)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} 训练平均损失: {avg_train_loss:.4f}")
        
        # 验证
        if (epoch+1)%validation_epochs==0:
            if accelerator.is_main_process:
                print("运行验证...")
            
            val_loss = validate(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                val_dataloader=val_loader,
                noise_scheduler=noise_scheduler,
                accelerator=accelerator,
            )
            
            if accelerator.is_main_process:
                print(f"Epoch {epoch+1} 验证损失: {val_loss:.4f}")
                
                # 记录到TensorBoard
                writer.add_scalar("val/loss", val_loss, epoch + 1)
                writer.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)

            if val_loss<best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter=0
                if accelerator.is_main_process:
                    best_model_path = os.path.join(output_dir,"best_model_ctc")
                    os.makedirs(best_model_path, exist_ok=True)
                    
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unwrapped_unet.save_pretrained(best_model_path)
                    # 保存最优模型的训练信息
                    best_info = {
                        "epoch": best_epoch,
                        "val_loss": best_val_loss,
                        "train_loss": avg_train_loss,
                        "n_gpus": n_gpus,
                        "effective_batch_size": batch_size * n_gpus * gradient_accumulation_steps
                    }
                    with open(os.path.join(best_model_path, "best_model_ctc_info.json"), "w") as f:
                        json.dump(best_info, f, indent=2)
                    
                    print(f"保存最优模型 (验证损失: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if accelerator.is_main_process:
                    print(f"验证损失未改善 ({patience_counter}/{early_stopping_patience})")
                
                if patience_counter >= early_stopping_patience:
                    if accelerator.is_main_process:
                        print(f"\n早停触发！最优模型在epoch {best_epoch}")
                    break
                    
   
        # 定期保存检查点
        if (epoch+1)%save_every_n_epochs==0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
                
                # Hugging Face Accelerate提供的小工具:
                # 把accelerator.prepare(...) 包装过的模型去掉加速包装,还原成原始的 PyTorch 模型
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(save_path)
                print(f"检查点已保存: {save_path}")
                
    # 保存最终模型
    if accelerator.is_main_process:
        print("\n" + "=" * 50)
        print("训练完成！保存最终模型...")
        print("=" * 50)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(output_dir,"final_model")
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(final_path)
        
        # 同时保存配置信息
        config_info = {
            "model_id": model_id,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "num_epochs": num_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": avg_train_loss,
            "n_gpus": n_gpus,
            "per_device_batch_size": batch_size,
            "effective_batch_size": batch_size * n_gpus * gradient_accumulation_steps
        }
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(config_info, f, indent=2)
        
        print(f"最终模型已保存到: {final_path}")
        print(f"最优模型已保存到: {os.path.join(output_dir,'best_model_ctc')}")
        print(f"最优epoch: {best_epoch}")
        print(f"最优验证损失: {best_val_loss:.4f}")
        
        writer.close()
        
        
        
    
if __name__ == "__main__":
    TRAIN_FILE_PATHS = [
        '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_1.csv',
        '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_2.csv',
        '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_3.csv',
    ]
    VAL_FILE_PATHS = ['/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_4.csv']
    
    
    print("\n"+"="*60)
    print("第一阶段: 全部细胞的lora训练")
    print("="*60)
    train_lora(
        train_csv_files=TRAIN_FILE_PATHS,
        val_csv_file = VAL_FILE_PATHS,
        output_dir='./stage1_all_cells_lora_peft',
        model_id="stabilityai/stable-diffusion-2-base",
        num_epochs=50, 
        batch_size=8,  # 根据显存调整
        val_batch_size=16,
        learning_rate=1e-4,
        gradient_accumulation_steps=4,
        lora_r=16,
        lora_alpha = 32,
        lora_dropout=0.1,
        save_every_n_epochs=20, 
        validation_epochs=5,
        early_stopping_patience=10,
    )
    print("第一阶段lora训练已完成")
    # ==================== 第二阶段 ====================
    print("\n" + "=" * 60)
    print("第二阶段：癌细胞专项LoRA微调")
    print("=" * 60)
    
    cancer_types = ["CTC"] 
    
    train_lora(
        train_csv_files=TRAIN_FILE_PATHS,
        val_csv_file=VAL_FILE_PATHS,
        output_dir=f"./stage2_cancer_lora_peft_{cancer_types}",
        model_id="stabilityai/stable-diffusion-2-base",
        pretrained_lora_path="./stage1_all_cells_lora_peft/best_model_ctc",  # 使用第一阶段最优模型
        filter_cell_type=cancer_types,
        num_epochs=100,
        batch_size=4,
        val_batch_size=8,
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        save_every_n_epochs=10,
        validation_epochs=5,
        early_stopping_patience=15
    )
    
    print("\n" + "=" * 60)
    print("第二次微调训练已完成！")
    print("=" * 60)