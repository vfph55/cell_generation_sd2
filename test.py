"""
使用PEFT微调后的SD2 LoRA进行癌细胞图生图数据扩增
"""

import pandas as pd
import os
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import cv2
import os,time

TARGET_SIZE = 128

from PIL import Image

def make_image_grid(images, rows=2, cols=3, cell_size=None, margin=8, bg=(255,255,255)):
    """
    images: PIL.Image 列表
    rows, cols: 网格行列
    cell_size: (w,h)。None 表示用每张图原始大小；若给定则会 resize 到该大小
    margin: 单元格间距（像素）
    bg: 背景色 (R,G,B)
    """
    assert len(images) > 0, "images 为空"
    n = min(len(images), rows * cols)

    # 统一尺寸
    if cell_size is None:
        cell_size = images[0].size  # (w,h)
    w, h = cell_size

    canvas_w = cols * w + (cols + 1) * margin
    canvas_h = rows * h + (rows + 1) * margin
    grid = Image.new("RGB", (canvas_w, canvas_h), color=bg)

    for i in range(n):
        r = i // cols
        c = i % cols
        x = margin + c * (w + margin)
        y = margin + r * (h + margin)

        img = images[i]
        if img.size != cell_size:
            img = img.resize(cell_size, Image.BICUBIC)
        grid.paste(img, (x, y))

    return grid


class CancerCellAugmentorPEFT:
    """基于PEFT的癌细胞图生图数据扩增器"""
    
    def __init__(
        self,
        lora_path="./stage2_cancer_lora_peft_[CTC]/best_model_ctc",
        model_id="stabilityai/stable-diffusion-2-base",
        device="cuda"
    ):
        print("加载模型...")
        
        # 加载基础pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # 加载PEFT LoRA适配器到UNet
        print(f"加载PEFT LoRA适配器: {lora_path}")
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            lora_path,
            is_trainable=False
        )
        
        # 使用更快的采样器
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(device)
        self.device = device
        
        print("✓ 模型加载完成")
    
    def pad_image_with_mean(self, image, target_size=TARGET_SIZE):
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
    
    def augment_single_image(
        self,
        image_path,
        cell_type,
        num_variations=5,
        strength=0.3,
        guidance_scale=7,
        num_inference_steps=50
    ):
        """
        对单张图像生成多个变体
        
        Args:
            image_path: 原始图像路径
            cell_type: 细胞类型
            num_variations: 生成变体数量
            strength: 变化强度 (0-1)，越大变化越大
            guidance_scale: 引导强度
            num_inference_steps: 推理步数
        
        Returns:
            生成的图像列表
        """
        # 读取原始图像
        init_image = Image.open(image_path).convert("RGB")
        print("是否是RGB图像:", init_image.mode)
        # 保存原图备份
        out_dir = "./test_augment_cancer_cell/images"
        os.makedirs(out_dir, exist_ok=True)
        original_path = os.path.join(out_dir, "original.png")
        init_image.save(original_path)
        print("shape of original image:", init_image.size)
        
        init_image = np.array(init_image)
        padded_img = self.pad_image_with_mean(init_image, TARGET_SIZE)

        
        # 生成prompt
        prompt = self._generate_prompt(cell_type)
        negative_prompt = "overexposed, blurry, low quality, distorted, artifacts, noisy, unclear, damaged"
        
        generated_images = []
        
        for i in range(num_variations):
            # 每次生成使用略微不同的参数
            var_strength = strength + np.random.uniform(-0.05, 0.05)
            var_strength = np.clip(var_strength, 0.3, 0.8)
            
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=padded_img,
                strength=var_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(i)
            ).images[0]
            
            generated_images.append(image)
        
        return generated_images
    
    def augment_dataset(
        self,
        csv_files,
        output_dir="./augmented_cancer_cells",
        cancer_types=None,
        num_variations_per_image=5,
        strength=0.5,
        conf_threshold=0.6
    ):
        """
        批量扩增整个数据集
        
        Args:
            csv_files: CSV文件列表
            output_dir: 输出目录
            cancer_types: 要扩增的癌细胞类型列表
            num_variations_per_image: 每张原图生成多少变体
            strength: 变化强度
            conf_threshold: 置信度阈值
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # 读取数据
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        
        # 过滤癌细胞
        if cancer_types:
            df = df[df['cell_type'].isin(cancer_types)]
        
        # 过滤置信度和存在的图像
        df = df[df['conf'] >= conf_threshold]
        df = df[df['image_file_path'].apply(os.path.exists)]
        
        print(f"找到 {len(df)} 张癌细胞图像待扩增")
        print(f"每张生成 {num_variations_per_image} 个变体")
        print(f"总共将生成 {len(df) * num_variations_per_image} 张新图像")
        
        # 扩增记录
        augmented_records = []
        
        # 批量处理
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="扩增进度"):
            image_path = row['image_file_path']
            cell_type = row['cell_type']
            
            try:
                # 生成变体
                generated_images = self.augment_single_image(
                    image_path=image_path,
                    cell_type=cell_type,
                    num_variations=num_variations_per_image,
                    strength=strength
                )
                
                # 保存生成的图像
                for var_idx, gen_image in enumerate(generated_images):
                    # 生成新文件名
                    original_filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(original_filename)
                    new_filename = f"{name}_aug_{var_idx}{ext}"
                    new_path = os.path.join(output_dir, "images", new_filename)
                    
                    # 保存
                    gen_image.save(new_path)
                    
                    # 记录
                    new_record = row.copy()
                    new_record['image_file_path'] = new_path
                    new_record['original_image'] = image_path
                    new_record['augmentation_id'] = var_idx
                    new_record['augmentation_strength'] = strength
                    augmented_records.append(new_record)
                    
            except Exception as e:
                print(f"\n处理 {image_path} 时出错: {e}")
                continue
        
        # 保存新的CSV
        augmented_df = pd.DataFrame(augmented_records)
        csv_output_path = os.path.join(output_dir, "augmented_dataset.csv")
        augmented_df.to_csv(csv_output_path, index=False)
        
        print(f"\n✓ 扩增完成!")
        print(f"  原始图像: {len(df)} 张")
        print(f"  生成图像: {len(augmented_df)} 张")
        print(f"  图像保存到: {os.path.join(output_dir, 'images')}")
        print(f"  CSV保存到: {csv_output_path}")
        
        return augmented_df
    
    def _generate_prompt(self, cell_type):
        """提示词制作(针对不同的细胞类型)"""
        prompt = f"a high-quality microscopic image of {cell_type} cell"
        type_ls = ['CD66b', 'CD14', 'WBC', 'CD3']
        
        if cell_type in type_ls:
            prompt = "a high-quality microscopic image of white blood cell, with regular nucleus morphology and smooth cell boundary"
        elif cell_type == 'CTC':
            prompt = f'A high-resolution microscopy image of a circulating tumour cell {cell_type}, large irregular nucleus, thin cytoplasm rim, irregular cell outline, sharp focus, detailed texture of cell membrane, realistic biomedical microscope photo style,dark background, high-contrast microscopy'
        elif cell_type == 'CHC':
            prompt = f'Brightfield micrograph of a cluster of circulating hybrid cells, mixed tumour/immune morphology, {cell_type}, irregular nuclei, abundant cytoplasm granules, realistic biomedical microscope photo style.'
        
        return prompt
    
    



if __name__ == "__main__":
    
    # ==================== 图生图扩增 ====================
    print("=" * 60)
    print("癌细胞图生图数据扩增")
    print("=" * 60)
    
    # 初始化扩增器（使用第二阶段PEFT最优模型）
    augmentor = CancerCellAugmentorPEFT(
        lora_path="./stage2_cancer_lora_peft_[CTC]/best_model_ctc",  # 使用验证集选出的最优模型
        model_id="stabilityai/stable-diffusion-2-base"
    )
    csv_files = [
        '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_1.csv',
        # '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_2.csv',
        # '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_3.csv',
        # '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_4.csv',
        # '/F00120250015/cell_datasets/dataset_zkw/test/251016/folds/fold_5.csv',
    ]
    
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    ctc_df = full_df[full_df['cell_type'].isin(['CTC'])]
    print("ctc ",len(ctc_df))

    paths = ctc_df['image_file_path'].tolist()
    
    strength = 0.6
    generated_imgs = augmentor.augment_single_image(paths[0],'CTC',num_variations=6,strength=strength)
    print("ganerated img shape", generated_imgs[0].size)
    
    out_dir = "./test_augment_cancer_cell/images"
    os.makedirs(out_dir, exist_ok=True)
    
    
    orig = os.path.basename(paths[0])
    stem, _ = os.path.splitext(orig)
    ts = time.strftime("%Y%m%d")  # 可选：给一批次加时间戳

    for i, img in enumerate(generated_imgs):
        save_path = os.path.join(out_dir, f"{stem}_aug_strenth_{strength}_{ts}_{i:02d}.png")
        img.save(save_path)
        print("saved:", save_path)
        
    grid = make_image_grid(generated_imgs, rows=2, cols=3, cell_size=generated_imgs[0].size,
                       margin=8, bg=(255,255,255))
    grid_path = os.path.join(out_dir, f"{stem}_aug_strenth_{strength}_{ts}_grid.png")
    grid.save(grid_path)
    print("saved grid:", grid_path)
    
    
    
