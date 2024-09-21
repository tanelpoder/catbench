# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2024 Tanel Poder
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

def compute_vit_embeddings(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"Warning: Cannot identify image file {image_path}. Skipping.")
        return None

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy()

def load_and_transform_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    except UnidentifiedImageError:
        print(f"Warning: Cannot identify image file {image_path}. Skipping.")
        return None

def compute_vit_embeddings_batch(image_paths):
    images = []
    valid_image_paths = []
    
    for image_path in image_paths:
        image = load_and_transform_image(image_path)
        if image is not None:
            images.append(image)
            valid_image_paths.append(image_path)

    if not images:
        return None, None

    images = torch.stack(images).to(device)

    with torch.no_grad():
        embeddings = model(images)
    
    return embeddings.cpu().numpy(), valid_image_paths


def main(image_dir, output_file, batch_size):
    
    image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for i in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[i:i + batch_size]
        embeddings, valid_image_paths = compute_vit_embeddings_batch(batch_image_paths)

        if embeddings is not None:
            for embedding, image_path in zip(embeddings, valid_image_paths):
                image_name = os.path.basename(image_path)
                append_embedding_to_file(output_file, image_name, embedding)
        else:
            print(f'Computing embeddings for batch starting at index {i} failed')
    
def append_embedding_to_file(file_path, image_name, embedding):
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    with open(file_path, 'a') as f:
        f.write(f"{image_name}\t{embedding_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ViT embeddings for all images in a directory.')
    parser.add_argument('image_dir', type=str, help='Directory containing images')
    parser.add_argument('output_file', type=str, help='File to store embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.eval().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    main(args.image_dir, args.output_file, batch_size=args.batch_size)

