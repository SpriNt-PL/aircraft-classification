import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import transforms

MODEL_PATH = "best_model.pth"  
CLASSES_PATH = "classes.json"
DATASET_DIR = "dataset/crop"  
IMG_SIZE = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Definition 
class CustomEfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0, global_pool="max"
        )
        num_features = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x


def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            return json.load(f)
    if os.path.isdir(DATASET_DIR):
        return sorted(
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        )
    raise FileNotFoundError(
        f"Neither '{CLASSES_PATH}' nor '{DATASET_DIR}' was found. "
        "Export classes.json from your notebook first."
    )


def load_model(num_classes):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint '{MODEL_PATH}' not found.")
    model = CustomEfficientNetB3(num_classes=num_classes).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Image transformation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Military Aircraft Classifier - Demo")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        self.photo_ref = None

        # Load classes and model
        try:
            self.classes = load_classes()
            self.model = load_model(len(self.classes))
        except Exception as e:
            messagebox.showerror("Startup Error", str(e))
            root.destroy()
            return

        # UI Elements
        tk.Label(
            root, text="Military Aircraft Classifier Demo", font=("Segoe UI", 16, "bold")
        ).pack(pady=(15, 5))

        tk.Button(
            root,
            text="Choose Photo",
            command=self.select_and_predict,
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="white",
            padx=18,
            pady=8,
            cursor="hand2",
        ).pack(pady=5)

        self.image_label = tk.Label(
            root,
            text="No image selected",
            width=45,
            height=18,
            bg="#f1f5f9",
            fg="#64748b",
            relief="solid",
            bd=1,
        )
        self.image_label.pack(pady=15)

        self.result_label = tk.Label(
            root, text="Predicted Tag: —", font=("Segoe UI", 15, "bold"), fg="#0f172a"
        )
        self.result_label.pack(pady=(5, 4))

        self.topk_label = tk.Label(
            root, text="", font=("Segoe UI", 10), fg="#475569", justify="center"
        )
        self.topk_label.pack(pady=4)

    def select_and_predict(self):
        filepath = filedialog.askopenfilename(
            title="Select an aircraft image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not open image:\n{e}")
            return

        # Display preview
        preview = image.copy()
        preview.thumbnail((340, 300))
        self.photo_ref = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.photo_ref, text="", width=340, height=300)

        # Run inference
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_idxs = torch.topk(probs, k=min(5, len(self.classes)))

        best_tag = self.classes[top_idxs[0].item()]
        best_conf = top_probs[0].item() * 100
        self.result_label.config(text=f"Predicted Tag: {best_tag} ({best_conf:.1f}%)")

        top5_lines = [
            f"{i+1}. {self.classes[idx.item()]} — {prob.item()*100:.1f}%"
            for i, (prob, idx) in enumerate(zip(top_probs, top_idxs))
        ]
        self.topk_label.config(text="Top 5 predictions:\n" + "\n".join(top5_lines))


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()