
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
import pandas as pd
from pathlib import Path
import random

from depth_anything_v2.dpt import DepthAnythingV2

# --- Configuration & Setup ---
def setup_logging(model_name):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"evaluation_{model_name}.log")
    
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger(__name__)

def setup_gpu():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset for Testing ---
class EndoscopeTestDataset(Dataset):
    def __init__(self, images_dir, gts_dir, resolution=(448, 448)):
        self.images_dir = images_dir
        self.gts_dir = gts_dir
        self.resolution = (resolution, resolution)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_basenames = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".png")}
        gt_basenames = {os.path.splitext(f)[0] for f in os.listdir(gts_dir) if f.endswith(".png")}
        self.common_files = sorted(list(image_basenames.intersection(gt_basenames)))

        if not self.common_files:
            raise RuntimeError("No matching image-ground_truth pairs found in the dataset directories.")
    
    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        basename = self.common_files[idx]
        img_path = os.path.join(self.images_dir, f"{basename}.png")
        gt_path = os.path.join(self.gts_dir, f"{basename}.png")

        image_orig = cv2.imread(img_path)
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        
        depth_gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
        
        image_resized = cv2.resize(image_orig, self.resolution, interpolation=cv2.INTER_LINEAR)
        image_tensor = self.transform(image_resized)
        
        depth_gt = depth_gt.astype(np.float32) * (300.0 / 65535.0)
        
        return {
            'image_tensor': image_tensor,
            'image_orig': image_orig,
            'depth_gt': depth_gt,
            'filename': f"{basename}.png"
        }

# --- Metrics & Alignment ---
def align_depth_scale_shift(pred, target):
    mask = target > 0
    if not mask.any(): return pred, 1.0, 0.0
    
    target_masked = target[mask]
    pred_masked = pred[mask]
    
    A = np.vstack([pred_masked, np.ones(len(pred_masked))]).T
    scale, shift = np.linalg.lstsq(A, target_masked, rcond=None)[0]
    
    return scale * pred + shift, scale, shift

def compute_depth_metrics(pred, target):
    # Clamp predictions to avoid log(0) issues and invalid divisions
    pred = np.maximum(pred, 1e-6)
    mask = target > 1e-6 # Use a small epsilon for the ground truth mask
    
    if not mask.any(): 
        return {m: np.nan for m in ["abs_rel","sq_rel","rmse","rmse_log","a1","a2","a3"]}

    pred_valid, target_valid = pred[mask], target[mask]
    
    thresh = np.maximum((target_valid / pred_valid), (pred_valid / target_valid))
    metrics = {
        'a1': (thresh < 1.25).mean(),
        'a2': (thresh < 1.25**2).mean(),
        'a3': (thresh < 1.25**3).mean(),
        'abs_rel': np.mean(np.abs(target_valid - pred_valid) / target_valid),
        'sq_rel': np.mean(((target_valid - pred_valid) ** 2) / target_valid),
        'rmse': np.sqrt(np.mean((target_valid - pred_valid) ** 2)),
        'rmse_log': np.sqrt(np.mean((np.log(target_valid) - np.log(pred_valid)) ** 2)),
    }
    return metrics

# --- Visualization ---
def visualize_comparison(sample, pred_orig_aligned, pred_ft, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    valid_gt = sample['depth_gt'][sample['depth_gt'] > 0]
    if len(valid_gt) == 0: return 
    vmin = np.min(valid_gt)
    vmax = np.max(valid_gt)

    axes[0].imshow(sample['image_orig'])
    axes[0].set_title('RGB Image')

    axes[1].imshow(sample['depth_gt'], cmap='plasma', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth')

    axes[2].imshow(pred_orig_aligned, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[2].set_title('Original (Aligned)')
    
    im = axes[3].imshow(pred_ft, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[3].set_title('Fine-tuned')

    for ax in axes: ax.axis('off')
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label='Depth (mm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, os.path.splitext(sample['filename'])[0] + '.png'), dpi=150)
    plt.close(fig)

# --- Main Evaluation Logic ---
def evaluate(args, logger):
    device = setup_gpu()
    logger.info(f"Starting evaluation for fine-tuned model: {args.finetuned_checkpoint}")
    
    model_configs = {
        's': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'b': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'l': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    try:
        model_type = args.finetuned_checkpoint.split('_')[1]
        model_config = model_configs[model_type]
    except (IndexError, KeyError):
        logger.error(f"Could not determine model type from checkpoint name: {args.finetuned_checkpoint}. Please ensure it follows the 'finetuned_s_best.pth' format.")
        return

    # Load Original Model
    original_model = DepthAnythingV2(**model_config)
    original_model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_vit{model_type}.pth", map_location='cpu'))
    original_model = original_model.to(device).eval()
    logger.info("Loaded original pre-trained model.")
    
    # Load Fine-tuned Model
    finetuned_model = DepthAnythingV2(**model_config)
    finetuned_model.load_state_dict(torch.load(args.finetuned_checkpoint, map_location='cpu'))
    finetuned_model = finetuned_model.to(device).eval()
    logger.info("Loaded fine-tuned model.")
    
    # Dataloader
    test_dataset = EndoscopeTestDataset(
        images_dir=os.path.join(args.data_root, "test", "images"),
        gts_dir=os.path.join(args.data_root, "test", "gts"),
        resolution=args.resolution
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples.")
    
    num_samples = len(test_dataset)
    indices_to_visualize = set(random.sample(range(num_samples), k=min(args.num_visualizations, num_samples)))
    logger.info(f"Will save {len(indices_to_visualize)} random visualizations.")
    
    results = []
    output_vis_dir = os.path.join(args.output_dir, f"visualizations_{model_type}")
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader, desc="Evaluating")):
            image_tensor = sample['image_tensor'].to(device)
            gt_depth_np = sample['depth_gt'].squeeze().numpy()
            
            # Get predictions
            pred_orig_raw = original_model(image_tensor).squeeze().cpu().numpy()
            pred_ft = finetuned_model(image_tensor).squeeze().cpu().numpy()
            
            h, w = gt_depth_np.shape
            pred_orig_raw = cv2.resize(pred_orig_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_ft = cv2.resize(pred_ft, (w,h), interpolation=cv2.INTER_LINEAR)

            pred_orig_aligned, _, _ = align_depth_scale_shift(pred_orig_raw, gt_depth_np)
            
            metrics_orig = compute_depth_metrics(pred_orig_aligned, gt_depth_np)
            metrics_ft = compute_depth_metrics(pred_ft, gt_depth_np)
            
            results.append({
                'filename': sample['filename'][0],
                **{f'orig_{k}': v for k, v in metrics_orig.items()},
                **{f'ft_{k}': v for k, v in metrics_ft.items()},
            })
            
            if i in indices_to_visualize:
                visualize_comparison(
                    {'image_orig': sample['image_orig'].squeeze().numpy(), 'depth_gt': gt_depth_np, 'filename': sample['filename'][0]},
                    pred_orig_aligned, pred_ft, output_vis_dir
                )
    
    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, f"metrics_{model_type}.csv"), index=False)
    
    avg_metrics = df.drop(columns=['filename']).mean()
    
    report = f"# Evaluation Report: {model_type.upper()} Model\n\n"
    report += "Comparison between the original pre-trained model (aligned to GT) and the fine-tuned model.\n\n"
    report += "| Metric     | Original (Aligned) | Fine-tuned | Improvement |\n"
    report += "|------------|--------------------|------------|-------------|\n"
    
    for metric in ["abs_rel", "sq_rel", "rmse", "rmse_log"]:
        orig_val = avg_metrics.get(f'orig_{metric}', np.nan)
        ft_val = avg_metrics.get(f'ft_{metric}', np.nan)
        improvement = (orig_val - ft_val) / orig_val * 100 if orig_val != 0 else 0
        report += f"| {metric.upper():<10} | {orig_val:.4f}             | **{ft_val:.4f}** | `{improvement:+.2f}%` |\n"
        
    for metric in ["a1", "a2", "a3"]:
        orig_val = avg_metrics.get(f'orig_{metric}', np.nan)
        ft_val = avg_metrics.get(f'ft_{metric}', np.nan)
        improvement = (ft_val - orig_val) / orig_val * 100 if orig_val != 0 else 0
        report += f"| δ < 1.25^{metric[-1]} | {orig_val:.4f}             | **{ft_val:.4f}** | `{improvement:+.2f}%` |\n"
        
    report_path = os.path.join(args.output_dir, f"report_{model_type}.md")
    with open(report_path, 'w') as f:
        f.write(report)
        
    logger.info(f"Evaluation complete. Report saved to {report_path}")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Depth Anything V2 model.")
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help="Path to the fine-tuned model checkpoint (e.g., 'checkpoints/finetuned_s_best.pth').")
    parser.add_argument('--data_root', type=str, default='pytorch_dataset_depth', help="Root directory of the split dataset.")
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help="Directory to save evaluation results.")
    parser.add_argument('--resolution', type=int, default=448, help="Square image resolution used during training.")
    parser.add_argument('--num_visualizations', type=int, default=20, help="Number of comparison images to save.")

    args = parser.parse_args()
    model_name_from_path = os.path.basename(args.finetuned_checkpoint).replace('.pth', '')
    logger = setup_logging(model_name_from_path)
    
    try:
        evaluate(args, logger)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)