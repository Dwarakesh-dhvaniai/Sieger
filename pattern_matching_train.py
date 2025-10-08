import cv2
import os
import shutil
import numpy as np
import tempfile
import yaml
from ultralytics import YOLO
# Standard library imports
import random
import shutil
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Callable
from itertools import combinations, product
import torchvision.transforms as T


# Third-party imports
import numpy as np
from PIL import Image

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# torchvision imports
import torchvision
from torchvision import transforms
from torchvision.io import decode_image
from torchvision.transforms import v2




def get_detections(dir_path):
    """
    Detects the largest outer contour and the most circular inner contour (if present)
    for all images in a given directory (including subfolders).

    Parameters:
        dir_path (str): Path to the directory containing input images.

    Returns:
        dict:
            A dictionary mapping image paths (relative to dir_path) to tuples of bounding boxes:
            {
                "subfolder/image1.jpg": [(x1, y1, w1, h1), (x2, y2, w2, h2)],
                "subfolder/image2.jpg": [(x1, y1, w1, h1), None],
                ...
            }
            - First tuple = outer bounding box.
            - Second tuple = inner bounding box (None if not found).

    Raises:
        FileNotFoundError: If the provided directory does not exist.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    results = {}

    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(img_path, dir_path)  # keep subfolder info
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Skipping {file}, could not read image.")
                    results[rel_path] = None
                    continue

                # ---- Preprocess image ----
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) < 1:
                    print(f"{file}: No contours found.")
                    results[rel_path] = None
                    continue

                # ---- Sort contours by area ----
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                outer = contours[0]
                x1, y1, w1, h1 = cv2.boundingRect(outer)

                # ---- Find best inner candidate ----
                inner_candidate = None
                best_score = 0

                for cnt in contours[1:]:  # skip outer contour
                    area = cv2.contourArea(cnt)
                    if area < 100:
                        continue

                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue

                    circularity = 4 * np.pi * (area / (perimeter * perimeter))

                    # Get center of contour
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Check if center lies inside outer contour
                    inside = cv2.pointPolygonTest(outer, (cx, cy), False)
                    if inside < 0:
                        continue

                    # Pick most circular one
                    if circularity > best_score and circularity > 0.7:
                        best_score = circularity
                        inner_candidate = cnt

                if inner_candidate is not None:
                    x2, y2, w2, h2 = cv2.boundingRect(inner_candidate)
                    results[rel_path] = [(x1, y1, w1, h1), (x2, y2, w2, h2)]
                else:
                    print(f"{file}: No inner circle found.")
                    results[rel_path] = [(x1, y1, w1, h1), None]

    return results

def save_cropped_from_detections(root_dir, save_dir, detections):
    import os, cv2
    cone_dir = os.path.join(save_dir, "cone_before_detection")
    pattern_dir = os.path.join(save_dir, "pattern_before_detection")
    os.makedirs(cone_dir, exist_ok=True)
    os.makedirs(pattern_dir, exist_ok=True)

    #detections = get_detections(root_dir)

    for rel_path, boxes in detections.items():
        if boxes is None:
            continue

        img_path = os.path.join(root_dir, rel_path)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping {img_path}, cannot read image.")
            continue

        outer_box, inner_box = boxes
        filename = os.path.basename(rel_path)  # just filename

        # --- Save outer circle (cone) → always flat folder ---
        if outer_box:
            x, y, w, h = outer_box
            cropped_outer = image[y:y+h, x:x+w]
            save_path_outer = os.path.join(cone_dir, filename)
            cv2.imwrite(save_path_outer, cropped_outer)
            print(f"Saved outer circle: {save_path_outer}")

        # --- Save inner circle (pattern) → preserve subfolders relative to root_dir ---
        if inner_box:
            # ensure paths are relative to root_dir
            subfolder = os.path.relpath(os.path.dirname(img_path), root_dir)
            pattern_subfolder = os.path.join(pattern_dir, subfolder)
            os.makedirs(pattern_subfolder, exist_ok=True)
            x, y, w, h = inner_box
            cropped_inner = image[y:y+h, x:x+w]
            save_path_inner = os.path.join(pattern_subfolder, filename)
            cv2.imwrite(save_path_inner, cropped_inner)
            print(f"Saved inner circle: {save_path_inner}")

# Example usage:
# save_cropped_from_detections("/home/dhvani23/dwarakesh/Combine/Data", "/home/dhvani23/dwarakesh/Combine/output")

def format_yolo_dataset_in_memory(input_dir, detections):
    """
    Converts bounding box detections into YOLO-compatible dataset format in-memory.
    Returns a list of dicts with 'image', 'labels', 'filename'.
    """
    dataset = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(root, file)
            rel_key = os.path.relpath(img_path, input_dir)
            boxes = detections.get(rel_key) or detections.get(file)
            if boxes is None:
                print(f"No detections found for {rel_key}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping {img_path}, cannot read image.")
                continue

            img_h, img_w = image.shape[:2]

            label_lines = []
            outer_box, inner_box = boxes

            if outer_box:
                x, y, w, h = outer_box
                class_id = 0
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            if inner_box:
                x, y, w, h = inner_box
                class_id = 1
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            dataset.append({
                'image': image,
                'labels': label_lines,
                'filename': file
            })

    print(f"In-memory YOLO dataset created with {len(dataset)} images.")
    return dataset


def split_yolo_dataset_in_memory(dataset, val_ratio=0.2):
    """
    Splits in-memory YOLO dataset into train/val.
    Returns a dict with 'train', 'val', 'nc', 'names'.
    """
    dataset_copy = dataset.copy()
    random.shuffle(dataset_copy)

    val_count = int(len(dataset_copy) * val_ratio)
    val_set = dataset_copy[:val_count]
    train_set = dataset_copy[val_count:]

    split_data = {
        "train": train_set,
        "val": val_set,
        "nc": 2,
        "names": ["outer", "inner"]
    }

    print(f"Dataset split complete! Train: {len(train_set)}, Val: {len(val_set)}")
    return split_data


def train_yolo_from_memory(split_data, model_name="yolov9t.pt", epochs=300):
    """
    Trains YOLO from an in-memory split dataset by using a temporary folder for images/labels.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_dir = os.path.join(tmp_dir, "train")
        val_dir = os.path.join(tmp_dir, "val")
        os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

        def save_subset(subset_data, subset_dir):
            for item in subset_data:
                img_path = os.path.join(subset_dir, "images", item['filename'])
                cv2.imwrite(img_path, item['image'])
                label_path = os.path.join(subset_dir, "labels", os.path.splitext(item['filename'])[0] + ".txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(item['labels']))

        save_subset(split_data['train'], train_dir)
        save_subset(split_data['val'], val_dir)

        # Create data.yaml
        yaml_path = os.path.join(tmp_dir, "data.yaml")
        data_yaml = {
            "train": os.path.join(train_dir, "images"),
            "val": os.path.join(val_dir, "images"),
            "nc": split_data['nc'],
            "names": split_data['names']
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        # Train YOLO
        print(f"Using dataset: {yaml_path}")
        print(f"Loading pretrained model: {model_name}")
        model = YOLO(model_name)
        print(f"Starting training for {epochs} epochs...")
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=768,
            batch=8,
            freeze="backbone",
            patience=50,
            device=0   # change to 'cpu' if no GPU
        )
        print("✅ Training complete! Check runs/detect/train folder for results.")


def run_yolo_on_dataset(model_path, input_dir, output_dir_cone, output_dir_pattern, conf_thresh=0.5):
    """
    Run YOLO model on dataset and save crops:
    - Outer circle (class 0) -> flat folder `output_dir_cone`
    - Inner circle (class 1) -> preserve subfolder structure in `output_dir_pattern`
    
    Args:
        model_path (str): Path to trained YOLO model.
        input_dir (str): Dataset with subfolders of images.
        output_dir_cone (str): Save cropped outer circle images here (flat folder).
        output_dir_pattern (str): Save cropped inner circle images here (subfolders preserved).
        conf_thresh (float): Minimum confidence threshold.
    """
    os.makedirs(output_dir_cone, exist_ok=True)
    os.makedirs(output_dir_pattern, exist_ok=True)
    
    model = YOLO(model_path)

    # Iterate over subfolders
    for subfolder, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(subfolder, file)
            rel_subfolder = os.path.relpath(subfolder, input_dir)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping {img_path}, cannot read image.")
                continue

            # Run YOLO inference
            results = model(img_path, conf=conf_thresh)

            # Each result corresponds to an image
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                cls_ids = result.boxes.cls.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(cls_ids[i])
                    cropped = image[y1:y2, x1:x2]

                    if cls_id == 0:  # outer circle
                        save_path = os.path.join(output_dir_cone, file)
                        cv2.imwrite(save_path, cropped)
                    elif cls_id == 1:  # inner circle
                        save_subfolder = os.path.join(output_dir_pattern, rel_subfolder)
                        os.makedirs(save_subfolder, exist_ok=True)
                        save_path = os.path.join(save_subfolder, file)
                        cv2.imwrite(save_path, cropped)

            print(f"Processed {img_path}")

    print("YOLO detection and cropping completed.")


# -----------------------------
# FewShotDatasetBuilder Class
# -----------------------------
class FewShotDatasetBuilder:
    """
    Builds a few-shot support set from a dataset of images organized in class subfolders.
    """

    def __init__(self, crops_root_dir: Path):
        self.crops_root_dir = crops_root_dir
        self.logger = logging.getLogger(__name__)

    def create_support_set(self, k: int, output_dir: Path) -> None:
        """
        Selects up to `k` images per class and copies them to `output_dir`.
        """
        # Get all class directories
        class_dirs = [d for d in self.crops_root_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            self.logger.error(f"No class directories found in {self.crops_root_dir}.")
            return

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each class
        for class_dir in class_dirs:
            class_id = class_dir.name
            image_paths = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

            if len(image_paths) == 0:
                print(f"❌ No images found in class '{class_id}' — skipping this class.")
                continue
            elif len(image_paths) < k:
                print(f"⚠️ Class '{class_id}' has only {len(image_paths)} images — using all available.")
                selected = image_paths
            else:
                selected = random.sample(image_paths, k)

            # Copy selected images to output folder
            saved_dir = output_dir / class_id
            saved_dir.mkdir(parents=True, exist_ok=True)
            for image_path in selected:
                shutil.copy2(image_path, saved_dir / image_path.name)

        print(f"✅ Support set created at: {output_dir}")


# -----------------------------
# Function wrapper
# -----------------------------
def build_few_shot_support_set(dataset_root: str, output_dir: str, k: int = 5) -> None:
    """
    Creates a few-shot support set from a dataset with class subfolders.

    Args:
        dataset_root (str): Path to the dataset root folder.
        output_dir (str): Path where the support set should be saved.
        k (int): Number of images per class to select (default 5).
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    dataset_root_path = Path(dataset_root)
    output_dir_path = Path(output_dir)

    builder = FewShotDatasetBuilder(crops_root_dir=dataset_root_path)
    builder.create_support_set(k=k, output_dir=output_dir_path)


class EpisodicTripletDatasetFromDir(Dataset):
    """
    PyTorch Dataset for episodic triplet training.

    Loads images organized in class subfolders and provides triplets
    (anchor, positive, negative) for each episode.

    Attributes:
        root_dir (Path): Root directory containing class subfolders.
        transform (Callable, optional): Transform to apply to images.
        episodes (int): Number of episodes (length of the dataset).
        class_to_images (dict): Mapping from class labels to image paths.
        image_list (list): List of tuples (class_label, image_path) for all images.
    """
    def __init__(self, root_dir: Union[str, Path], transform: Optional[Callable] = None, episodes: int = 1000) -> None:
        """
        Initialize the dataset by scanning class subfolders and collecting image paths.

        Args:
            root_dir (str or Path): Path to the root directory containing class subfolders.
            transform (Callable, optional): Optional transform to apply to images.
            episodes (int): Number of episodes to generate; determines dataset length.

        Raises:
            ValueError: If no images are found in any class subfolder.
        """
        self.root_dir = Path(root_dir) if not isinstance(root_dir, Path) else root_dir
        self.transform = transform
        self.episodes = episodes
        self.class_to_images = {}  # Mapping from class label to list of image paths
        self.image_list: List[Tuple[str, Path]] = []  # List of tuples: (class_label, image_path)
        
        for class_folder in self.root_dir.iterdir():
            if class_folder.is_dir():
                class_label = class_folder.name
                images = list(class_folder.glob("*"))
                if images:
                    self.class_to_images[class_label] = images
                    for img_path in images:
                        self.image_list.append((class_label, img_path))
                else:
                    print(f"⚠️ Warning: No images found in class folder {class_folder}")
        
        if not self.image_list:
            raise ValueError(f"No images found in any subfolder of {self.root_dir}")
            
    def __len__(self) -> int:
        """
        Return the number of episodes in the dataset.

        Returns:
            int: Number of episodes.
        """
        return self.episodes
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet (anchor, positive, negative) for the given episode index.

        Randomly samples:
            - Anchor image from the dataset.
            - Positive image from the same class (different from anchor if possible).
            - Negative image from a different class.

        Applies the optional transform to all images before returning.

        Args:
            idx (int): Index of the episode (not used for sampling, only for dataset length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Anchor, positive, and negative images as tensors.

        Raises:
            ValueError: If there is only one class in the dataset (cannot sample negative).
        """
        # Randomly sample an anchor from the full list
        anchor_label, anchor_path = random.choice(self.image_list)



        # anchor_img = decode_image(str(anchor_path))
        anchor_img = decode_image_from_path(str(anchor_path)) # C,H,W and uint8

        
        # Choose a positive sample from the same class, ensuring it's different if possible
        positive_candidates = self.class_to_images[anchor_label].copy()
        if anchor_path in positive_candidates:
            positive_candidates.remove(anchor_path)
        if positive_candidates:
            positive_path = random.choice(positive_candidates)
        else:
            positive_path = anchor_path
            print(f"⚠️ Only one image in class '{anchor_label}'. Using anchor as positive.")
        # positive_img = decode_image(str(positive_path))
        positive_img = decode_image_from_path(str(positive_path))
        
        # Choose a negative sample from a different class
        negative_classes = [cls for cls in self.class_to_images.keys() if cls != anchor_label]
        if not negative_classes:
            raise ValueError("Only one class available in dataset; cannot sample negative.")
        negative_class = random.choice(negative_classes)
        negative_path = random.choice(self.class_to_images[negative_class])
        # negative_img = decode_image(str(negative_path))
        negative_img = decode_image_from_path(str(negative_path))
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img
    

## helper function to decode image from path
def decode_image_from_path(image_path):
    """
    Load an image from a file path and convert it to RGB format.

    Args:
        image_path (str or Path): Path to the image file.

    Returns:
        PIL.Image.Image: The image loaded as a PIL Image in RGB format.
    """
    # Open with OpenCV (BGR)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    return Image.fromarray(img)



class TripletNet(nn.Module):
    """
    Triplet Network for learning embeddings using triplet loss.

    The network takes three inputs: anchor, positive, and negative images,
    and computes their embeddings using a shared embedding network. It also
    calculates the pairwise distances between anchor-positive and anchor-negative embeddings.

    Attributes:
        embedding_net (nn.Module): The shared embedding network.
    """
    def __init__(self, embedding_net):
        """
        Initialize the TripletNet with a given embedding network.

        Args:
            embedding_net (nn.Module): Network to compute embeddings for input images.
        """

        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, a, p, n):
        """
        Forward pass to compute embeddings and pairwise distances.

        Args:
            a (torch.Tensor): Anchor images tensor.
            p (torch.Tensor): Positive images tensor.
            n (torch.Tensor): Negative images tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - dist_AP: Pairwise L2 distance between anchor and positive embeddings.
                - dist_AN: Pairwise L2 distance between anchor and negative embeddings.
                - embedded_A: Anchor embeddings.
                - embedded_P: Positive embeddings.
                - embedded_N: Negative embeddings.
        """

        embedded_A = self.embedding_net(a)
        embedded_P = self.embedding_net(p)
        embedded_N = self.embedding_net(n)

        dist_AP = F.pairwise_distance(embedded_A, embedded_P, 2) # 2 is for L2 norm
        dist_AN = F.pairwise_distance(embedded_A, embedded_N, 2)

        return dist_AP, dist_AN, embedded_A, embedded_P, embedded_N

class EmbeddingNet(nn.Module):
    """
    Convolutional neural network to compute embeddings from images.

    Uses a ResNet-50 backbone (pretrained on ImageNet optionally) and
    replaces the final classification layer with a fully connected
    network to produce embeddings of a specified size.

    Attributes:
        embedding_size (int): Dimension of the output embedding vector.
        convnet (nn.Module): ResNet-50 backbone without the final classification layer.
        fc (nn.Sequential): Fully connected layers mapping ResNet features to embeddings.
    """

    def __init__(self, embedding_size: int, use_pretrained: bool=True):
        """
        Initialize the embedding network.

        Args:
            embedding_size (int): Dimension of the output embedding vector.
            use_pretrained (bool, optional): Whether to use ImageNet pretrained weights. Defaults to True.
        """
        super(EmbeddingNet, self).__init__()
        self.embedding_size = embedding_size
        if use_pretrained:
            self.convnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        else:
            self.convnet = torchvision.models.resnet50(weights=None)

        # Get the input dimension of last layer
        self.fc_in_features = self.convnet.fc.in_features
        
        # Remove the last layer
        self.convnet = nn.Sequential(*list(self.convnet.children())[:-1])

        # Add linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.embedding_size)   
        )

    def forward(self, x):
        """
        Forward pass through the embedding network.

        Args:
            x (torch.Tensor): Input images tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_size).
        """
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FewShotTrainer:
    """
    A class to train few-shot learning models using triplet loss.
    """
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, output_dir: Path):
        """
        Initializes the FewShotTrainer.

        Args:
            model (nn.Module): The triplet network model to train.
            train_dataloader (DataLoader): DataLoader for the training set.
            val_dataloader (DataLoader): DataLoader for the validation set.
            output_dir (Path): Directory to save training outputs.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, epochs: int, criterion: nn.Module, optimizer: optim.Optimizer) -> None:
        """
        Trains the model for a specified number of epochs.

        Args:
            epochs (int): The number of training epochs.
            criterion (nn.Module): The loss function (e.g., TripletLoss).
            optimizer (optim.Optimizer): The optimization algorithm (e.g., Adam).
        """
        best_val_accuracy = 0
        train_losses: List[float] = []
        train_accuracies: List[float] = []
        val_losses: List[float] = []
        val_accuracies: List[float] = []

        logging.info(f"Starting training for {epochs} epochs on {self.device}.")

        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_accuracy = self._train_one_epoch(criterion, optimizer)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            val_loss, val_accuracy = self._test_one_epoch(criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            self._save_model(epoch, optimizer)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_best_model(optimizer)

        self._save_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        logging.info("Training finished.")

    def _train_one_epoch(self, criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Performs one training epoch.

        Args:
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer.

        Returns:
            Tuple[float, float]: The average training loss and accuracy for the epoch.
        """
        self.model.train()
        epoch_loss = 0
        epoch_correct = 0

        for A_tensors, P_tensors, N_tensors in self.train_dataloader:
            A_tensors, P_tensors, N_tensors = A_tensors.to(self.device), P_tensors.to(self.device), N_tensors.to(self.device)

            dist_AP, dist_AN, embedded_A, embedded_P, embedded_N = self.model(A_tensors, P_tensors, N_tensors)
            loss = criterion(embedded_A, embedded_P, embedded_N)  # Assuming TripletLoss expects distances and margin
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = (dist_AP < dist_AN).sum().item()
            epoch_correct += correct

        avg_loss = epoch_loss / len(self.train_dataloader)
        avg_accuracy = epoch_correct / len(self.train_dataloader.dataset)
        return avg_loss, avg_accuracy

    def _test_one_epoch(self, criterion: nn.Module) -> Tuple[float, float]:
        """
        Performs one validation/testing epoch.

        Args:
            criterion (nn.Module): The loss function.

        Returns:
            Tuple[float, float]: The average validation/testing loss and accuracy for the epoch.
        """
        self.model.eval()
        epoch_loss = 0
        epoch_correct = 0

        with torch.no_grad():
            for A_tensors, P_tensors, N_tensors in self.val_dataloader:
                A_tensors, P_tensors, N_tensors = A_tensors.to(self.device), P_tensors.to(self.device), N_tensors.to(self.device)

                dist_AP, dist_AN, embedded_A, embedded_P, embedded_N = self.model(A_tensors, P_tensors, N_tensors)
                loss = criterion(embedded_A, embedded_P, embedded_N) # Assuming TripletLoss expects distances and margin
                epoch_loss += loss.item()

                correct = (dist_AP < dist_AN).sum().item()
                epoch_correct += correct

        avg_loss = epoch_loss / len(self.val_dataloader)
        avg_accuracy = epoch_correct / len(self.val_dataloader.dataset)
        return avg_loss, avg_accuracy

    def _save_best_model(self, optimizer: optim.Optimizer) -> None:
        """Saves the best model state, optimizer state, and entire model."""
        torch.save(self.model.state_dict(), self.output_dir / "best_fewshot_model_state_dict.pth")
        torch.save(optimizer.state_dict(), self.output_dir / "best_fewshot_optimizer_state_dict.pth")
        torch.save(self.model, self.output_dir / "best_fewshot_model.pth")
        logging.info("Best model saved.")

    def _save_model(self, epoch: int, optimizer: optim.Optimizer) -> None:
        """Saves the model state, optimizer state, and entire model for the current epoch."""
        torch.save(self.model.state_dict(), self.output_dir / f"fewshot_model_state_dict_epoch_{epoch}.pth")
        torch.save(optimizer.state_dict(), self.output_dir / f"fewshot_optimizer_state_dict_epoch_{epoch}.pth")
        torch.save(self.model, self.output_dir / f"fewshot_model_epoch_{epoch}.pth")
        logging.info(f"Model saved at epoch {epoch+1}.")

    def _save_training_history(self, train_losses: List[float], train_accuracies: List[float], val_losses: List[float], val_accuracies: List[float]) -> None:
        """Saves the training history to a pickle file."""
        history = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
        with open(self.output_dir / "training_history.pkl", "wb") as f:
            pickle.dump(history, f)
        logging.info("Training history saved.")
    
    def evaluate_best_model(self, similarity_threshold: float = 0.9):
        """
        Evaluate the best saved model on the test dataset by comparing test images
        with training images. Computes accuracy, precision, recall, and F1-score.
        """
        device = self.device
        best_model_path = self.output_dir / "best_fewshot_model.pth"

        # Load best model
        checkpoint = torch.load(best_model_path, map_location=device)
        embedding_net = self.model.embedding_net.__class__(
            embedding_size=self.model.embedding_net.fc[-1].out_features
        )
        model = TripletNet(embedding_net)
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        embedding_model = model.embedding_net
        embedding_model.eval()

        # Transform
        transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Compute embeddings for train images
        train_embeddings_dict = {}
        for label, img_paths in self.train_dataloader.dataset.class_to_images.items():
            train_embeddings_dict[label] = []
            for img_path in img_paths:
                img = decode_image_from_path(str(img_path))
                img = transform(img)
                img_tensor = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = embedding_model(img_tensor)
                train_embeddings_dict[label].append(emb.cpu())

        # Compute embeddings for test images
        test_embeddings_dict = {}
        for label, img_paths in self.val_dataloader.dataset.class_to_images.items():
            test_embeddings_dict[label] = []
            for img_path in img_paths:
                img = decode_image_from_path(str(img_path))
                img = transform(img)
                img_tensor = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = embedding_model(img_tensor)
                test_embeddings_dict[label].append(emb.cpu())

        # Compare test images with train images
        y_true, y_pred = [], []

        # Same-class comparisons
        for label, test_emb_list in test_embeddings_dict.items():
            train_emb_list = train_embeddings_dict[label]
            for test_emb in test_emb_list:
                for train_emb in train_emb_list:
                    cos_sim = F.cosine_similarity(test_emb, train_emb, dim=1).item()
                    y_true.append(1)
                    y_pred.append(1 if cos_sim > similarity_threshold else 0)

        # Different-class comparisons
        train_labels = list(train_embeddings_dict.keys())
        for test_label, test_emb_list in test_embeddings_dict.items():
            for train_label in train_labels:
                if train_label == test_label:
                    continue
                for test_emb in test_emb_list:
                    for train_emb in train_embeddings_dict[train_label]:
                        cos_sim = F.cosine_similarity(test_emb, train_emb, dim=1).item()
                        y_true.append(0)
                        y_pred.append(1 if cos_sim > similarity_threshold else 0)

        # Metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = (y_true == y_pred).mean()  # ✅ Add this line
        precision = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1)
        recall = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

        logging.info("===== Best Model Metrics =====")
        logging.info(f"✅ Accuracy: {accuracy:.4f}")  # ✅ Add this line
        logging.info(f"✅ Precision: {precision:.4f}")
        logging.info(f"✅ Recall: {recall:.4f}")
        logging.info(f"✅ F1-score: {f1:.4f}")

def train_fewshot_triplet(
    dataset_root: str,
    output_dir: str,
    embedding_size: int = 64,
    num_episodes: int = 6400,
    train_val_split: list = [0.8, 0.2],
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 0.0005,
    margin: float = 1.0,
    k: int = 5  # <-- number of images per class
):

    """
    Train a few-shot triplet network on a dataset.

    Args:
        dataset_root (str): Root folder containing the dataset.
        output_dir (str or Path): Directory to save model checkpoints and outputs.
        embedding_size (int): Dimension of the embedding vector.
        num_episodes (int): Total number of triplet episodes to generate.
        train_val_split (list): Fraction of episodes for train and validation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        margin (float): Margin for TripletMarginLoss.
        k (int): Number of images per class to select for few-shot training.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Build few-shot support set from dataset_root
    support_set_dir = output_dir / "support_set"
    build_few_shot_support_set(dataset_root, support_set_dir, k=k)

    # Compute number of episodes
    num_train_episodes = int(num_episodes * train_val_split[0])
    num_val_episodes = int(num_episodes * train_val_split[1])

    # -----------------------------
    # 2️⃣ Transform
    # -----------------------------
    apn_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # 3️⃣ Dataset & DataLoader
    # -----------------------------
    train_dataset = EpisodicTripletDatasetFromDir(
        root_dir=support_set_dir,
        transform=apn_transform,
        episodes=num_train_episodes
    )
    val_dataset = EpisodicTripletDatasetFromDir(
        root_dir=support_set_dir,
        transform=apn_transform,
        episodes=num_val_episodes
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 4️⃣ Model
    # -----------------------------
    embedding_net = EmbeddingNet(embedding_size=embedding_size, use_pretrained=True)
    model = TripletNet(embedding_net)

    # -----------------------------
    # 5️⃣ Loss & Optimizer
    # -----------------------------
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------
    # 6️⃣ Trainer
    # -----------------------------
    trainer = FewShotTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=output_dir
    )

    # Train
    trainer.train(epochs=epochs, criterion=criterion, optimizer=optimizer)
    trainer.evaluate_best_model()
    print(f"Training completed. Models saved at {output_dir}")

