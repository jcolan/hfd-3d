# ml_tools/2_train.py 
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
import logging
from tqdm import tqdm
import argparse

# Now, this import will work correctly
from depth_anything_v2.dpt import DepthAnythingV2

# --- Configuration & Setup ---
def setup_logging(model_name):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"training_{model_name}.log")

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
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for training.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    return device

# --- Dataset Class ---
class EndoscopeDataset(Dataset):
    def __init__(self, images_dir, gts_dir, transform=None, resolution=(448, 448)):
        self.images_dir = images_dir
        self.gts_dir = gts_dir
        self.transform = transform
        self.resolution = resolution
        self.target_height, self.target_width = resolution

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(gts_dir):
            raise FileNotFoundError(f"Ground truth directory not found: {gts_dir}")

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
        self.gt_files = sorted([f for f in os.listdir(gts_dir) if f.endswith(".png")])

        if not self.image_files:
            raise RuntimeError(f"No PNG images found in {images_dir}")
        if not self.gt_files:
            raise RuntimeError(f"No PNG images found in {gts_dir}")

        if len(self.image_files) != len(self.gt_files):
            raise RuntimeError(f"Mismatched number of files. Images: {len(self.image_files)}, GTs: {len(self.gt_files)}")

        for i in range(min(5, len(self.image_files))):
            img_name = os.path.splitext(self.image_files[i])[0]
            gt_name = os.path.splitext(self.gt_files[i])[0]
            if img_name != gt_name:
                raise RuntimeError(f"Filename mismatch detected: {self.image_files[i]} vs {self.gt_files[i]}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_path = os.path.join(self.gts_dir, self.gt_files[idx])
        depth_gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)
        if depth_gt is None:
            raise ValueError(f"Failed to load depth map: {gt_path}")

        image = cv2.resize(image, self.resolution, interpolation=cv2.INTER_LINEAR)
        depth_gt = cv2.resize(depth_gt, self.resolution, interpolation=cv2.INTER_NEAREST)

        depth_gt = depth_gt.astype(np.float32) * (300.0 / 65535.0)

        if self.transform:
            image = self.transform(image)

        depth_gt = torch.from_numpy(depth_gt).unsqueeze(0)

        return image, depth_gt

# --- Loss Function ---
class DepthLoss(nn.Module):
    def __init__(self, lambda_si=0.5):
        super(DepthLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambda_si = lambda_si

    def forward(self, pred, target):
        valid_mask = target > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_masked = pred[valid_mask]
        target_masked = target[valid_mask]

        l1 = self.l1_loss(pred_masked, target_masked)
        log_diff = torch.log(pred_masked) - torch.log(target_masked)
        si_loss = torch.sqrt(torch.mean(log_diff**2) - (torch.mean(log_diff)) ** 2)

        return l1 + self.lambda_si * si_loss

# --- Training Loop ---
def train_model(args, logger):
    device = setup_gpu()
    logger.info(f"Starting training for model: {args.model_type}")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model_configs = {
        's': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'b': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'l': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    model_config = model_configs[args.model_type]
    model = DepthAnythingV2(**model_config)

    checkpoint_path = f"checkpoints/depth_anything_v2_vit{args.model_type}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pre-trained weights not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    logger.info(f"Loaded pre-trained weights from {checkpoint_path}")

    if args.grad_checkpointing:
        model.pretrained.blocks.use_checkpoint = True
        logger.info("Gradient checkpointing enabled.")

    model = model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = EndoscopeDataset(
        images_dir=os.path.join(args.data_root, "train", "images"),
        gts_dir=os.path.join(args.data_root, "train", "gts"),
        transform=transform,
        resolution=(args.resolution, args.resolution)
    )
    val_dataset = EndoscopeDataset(
        images_dir=os.path.join(args.data_root, "val", "images"),
        gts_dir=os.path.join(args.data_root, "val", "gts"),
        transform=transform,
        resolution=(args.resolution, args.resolution)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    logger.info(f"Training with {len(train_dataset)} images, validating with {len(val_dataset)} images.")
    logger.info(f"Resolution: {args.resolution}x{args.resolution}, Batch Size: {args.batch_size}, Accumulation Steps: {args.accum_steps}")
    logger.info(f"Effective Batch Size: {args.batch_size * args.accum_steps}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = DepthLoss().to(device)
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, (images, depths) in enumerate(pbar_train):
            images, depths = images.to(device), depths.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, depths) / args.accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * args.accum_steps
            pbar_train.set_postfix({'loss': f"{loss.item() * args.accum_steps:.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, depths in pbar_val:
                images, depths = images.to(device), depths.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, depths)
                val_loss += loss.item()
                pbar_val.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_save_path = f"checkpoints/finetuned_{args.model_type}_best.pth"
            torch.save(model.state_dict(), checkpoint_save_path)
            logger.info(f"New best model saved to {checkpoint_save_path} with Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Depth Anything V2 model.")
    parser.add_argument('--model_type', type=str, required=True, choices=['s', 'b', 'l'], help="Model type: s, b, or l.")
    parser.add_argument('--data_root', type=str, default='pytorch_dataset_depth', help="Root directory of the split dataset.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per GPU.")
    parser.add_argument('--accum_steps', type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate.")
    parser.add_argument('--resolution', type=int, default=448, help="Square image resolution for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--grad_checkpointing', action='store_true', help="Enable gradient checkpointing for memory saving.")

    args = parser.parse_args()
    logger = setup_logging(args.model_type)

    try:
        train_model(args, logger)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)