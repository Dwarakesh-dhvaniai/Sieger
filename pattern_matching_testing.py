import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/home/dhvani23/dwarakesh/Combine/fewshot_training_output/best_fewshot_model.pth"   # your trained model
TRAIN_FOLDER = "/home/dhvani23/dwarakesh/Combine/images/train"           # folder with 5 support images
TEST_IMAGE = "/home/dhvani23/dwarakesh/Combine/images/testing/3733.png"  # single test image

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# LOAD MODEL
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(MODEL_PATH, map_location=device)
model.eval()  # evaluation mode

# -----------------------------
# HELPER: IMAGE TO EMBEDDING
# -----------------------------
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.embedding_net(img)  # use embedding network
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding
# -----------------------------
# GET EMBEDDINGS
# -----------------------------
# Train embeddings
train_embeddings = {}
for fname in os.listdir(TRAIN_FOLDER):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(TRAIN_FOLDER, fname)
        train_embeddings[fname] = get_embedding(path)

# Test embedding
test_embedding = get_embedding(TEST_IMAGE)

# -----------------------------
# FIND BEST MATCH
# -----------------------------
best_match = None
best_score = -1  # cosine similarity range [-1,1]

for fname, emb in train_embeddings.items():
    sim = F.cosine_similarity(test_embedding, emb)
    score = sim.item()
    print(f"Similarity with {fname}: {score:.4f}")
    if score > best_score:
        best_score = score
        best_match = fname

print(f"\nBest match for test image: {best_match} (score: {best_score:.4f})")
