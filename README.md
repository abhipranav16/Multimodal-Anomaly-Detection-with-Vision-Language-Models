# 📌 Multimodal Anomaly Detection with Vision-Language Models

**Author:** _Your Name_  
**Course:** DA623 – Winter 2025

---

## 💡 Motivation

Anomaly detection is essential in fields like manufacturing, healthcare, and cybersecurity. However, traditional systems usually rely on a single modality—such as image-based inspection or log analysis—which can miss critical signals when data is noisy or incomplete.

This project explores the use of **vision-language models**, particularly CLIP, to detect anomalies by comparing the **semantic similarity between images and their corresponding textual descriptions**. If the visual and textual inputs don’t align, the sample may be considered anomalous. This approach offers an intuitive and generalizable way to flag mismatches in multimodal data.

---

## 🧠 Background: Multimodal Learning and CLIP

Multimodal learning combines different types of inputs (e.g., text + image) to create richer and more robust representations. Vision-language models like **CLIP (Contrastive Language–Image Pretraining)** and **BLIP** have revolutionized this space by learning a **shared embedding space** for image-text pairs.

CLIP was trained on 400M image-caption pairs to align visual and textual information. It's primarily used for **zero-shot classification, retrieval, and captioning**, but here, we creatively apply it to detect **semantic mismatches**, which signal anomalies.

---

## 🔍 Problem Formulation

Given:
- An image `I` (e.g., a product or medical scan)
- A textual description `T` (e.g., log entry, label, or caption)

Goal:
- Encode both `I` and `T` using a vision-language model like CLIP.
- Compute the **cosine similarity** between the two embeddings.
- Flag samples with **low similarity** as potential anomalies.

This method is unsupervised, flexible, and generalizes across domains.

---

## 🧪 Code Demo

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image and define candidate text descriptions
image = Image.open("sample_image.jpg")
texts = ["a clean metal nut", "a damaged metal nut", "a broken piece"]

# Encode inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Calculate similarity
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Print results
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.3f}")
