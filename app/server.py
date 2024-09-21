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

import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
import numpy as np
from urllib.parse import urlparse, parse_qs, unquote

HOST = '0.0.0.0'
PORT = 8000

APP_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR    = os.path.join(APP_DIR, 'embeddings')
IMAGE_ROOT = os.path.join(APP_DIR, 'data')
PLANE_DIR  = os.path.join(IMAGE_ROOT, 'fgvc-aircraft-2013b/data/images/')
CAT_DIR    = os.path.join(IMAGE_ROOT, 'PetImages/Cat/')
DOG_DIR    = os.path.join(IMAGE_ROOT, 'PetImages/Dog/')

def parse_line(line):
    filename, embedding_str = line.strip().split('\t', 1)
    embedding = np.fromstring(embedding_str.strip()[1:-1], sep=',')
    return filename, embedding

# Compress the column value ranges for more prominent visualization for humans
def normalize_column(column):
    min_val = np.min(column)
    max_val = np.max(column)
    if min_val == max_val:  # Avoid division by zero
        return np.zeros_like(column)
    else:
        return (column - min_val) / (max_val - min_val)

datasets = {
    'planes': {
        'embedding_file': os.path.join(EMB_DIR, 'plane_embeddings_small.tsv'),
        'image_dir': PLANE_DIR,
        'embeddings': [],
        'file_names': [],
        'norm_embeddings': None,
    },
    'cats': {
        'embedding_file': os.path.join(EMB_DIR, 'cat_embeddings_small.tsv'),
        'image_dir': CAT_DIR,
        'embeddings': [],
        'file_names': [],
        'norm_embeddings': None,
    },
    'dogs': {
        'embedding_file': os.path.join(EMB_DIR, 'dog_embeddings_small.tsv'),
        'image_dir': DOG_DIR,
        'embeddings': [],
        'file_names': [],
        'norm_embeddings': None,
    },
}

# Load embeddings for .tsv files for each dataset
for dataset_name, dataset_info in datasets.items():
    embeddings = []
    file_names = []
    embedding_file = dataset_info['embedding_file']
    embedding_file_path = os.path.join('.', embedding_file)  # Adjust path if needed
    if not os.path.isfile(embedding_file_path):
        print(f"Embedding file for {dataset_name} not found: {embedding_file_path}")
        continue
    with open(embedding_file_path, 'r') as f:
        for line in f:
            filename, embedding = parse_line(line)
            file_names.append(filename)
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    dataset_info['embeddings'] = embeddings
    dataset_info['file_names'] = file_names
    dataset_info['norm_embeddings'] = np.apply_along_axis(normalize_column, 0, embeddings)

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        if path == '/data':
            is_sorted = query_params.get('sorted', ['0'])[0] == '1'
            is_normalized = query_params.get('normalized', ['0'])[0] == '1'
            dataset_name = query_params.get('dataset', [None])[0]
            if dataset_name not in datasets:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid dataset parameter')
                return
            dataset_info = datasets[dataset_name]
            embeddings = dataset_info['embeddings']
            file_names = dataset_info['file_names']
            norm_embeddings = dataset_info['norm_embeddings']

            data_to_send = []
            for i in range(len(file_names)):
                embedding = embeddings[i]
                if is_normalized:
                    embedding = norm_embeddings[i]
                if is_sorted:
                    embedding = np.sort(embedding)
                data_to_send.append({
                    "filename": file_names[i],
                    "embedding": embedding.tolist()
                })

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data_to_send).encode('utf-8'))

        elif path == '/heatmap':
            dataset_name = query_params.get('dataset', [None])[0]
            if dataset_name not in datasets:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid dataset parameter')
                return

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('heatmap.html', 'r') as f:
                html_content = f.read()
                self.wfile.write(html_content.encode('utf-8'))

        elif path.startswith('/image/'):
            # Path format: /image/<dataset>/<image_path>
            path_parts = path.split('/')
            if len(path_parts) < 3:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid image path')
                return
            dataset_name = path_parts[2]
            if dataset_name not in datasets:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'Invalid dataset')
                return
            image_rel_path = '/'.join(path_parts[3:])
            image_name = unquote(image_rel_path)
            image_dir = datasets[dataset_name]['image_dir']
            image_path = os.path.normpath(os.path.join(image_dir, image_name))

            # Ensure that image_path is within image_dir
            if not image_path.startswith(os.path.abspath(image_dir)):
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b'Forbidden')
                return
            if os.path.isfile(image_path):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                with open(image_path, 'rb') as image_file:
                    self.wfile.write(image_file.read())
            else:
                self.send_response(404)
                self.end_headers()

        elif path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'r') as f:
                self.wfile.write(f.read().encode('utf-8'))
        else:
            # For other requests, serve files from the filesystem (not very secure!)
            super().do_GET()

with HTTPServer((HOST, PORT), CustomHandler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()
