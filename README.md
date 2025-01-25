# catbench
Cat Benchmarking at Scale, finally!

In this prerelease (v0.2) version I'm publishing a very simple embedding vector visualization app that plots out embeddings computed from various cat, dog and plane photos as a heatmap. Everything is precomputed and stored in text files, so you don't need PyTorch, GPUs or even a database.

## Announouncement blog entry

* https://tanelpoder.com/posts/visualizing-embedding-vectors-as-heatmaps/

![Cats Dogs Planes](/landing/cat-dog-plane-embeddings-heatmap-annotated.png)

## Installation Steps

25000 cat/dog images are included in this repository. If you want to download aircraft images too, use the `wget` command below. I have tested this on RHEL9 and Ubuntu 24.04 so far. You need to have python and pip installed in your OS for this.

Download and set up CatBench:

```
git clone https://github.com/tanelpoder/catbench
cd catbench

pip install -r requirements.txt

# if you want airplane images
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar xf fgvc-aircraft-2013b.tar.gz

# run the app
cd ../catvector
python catvector.py
```
Then go to `hostname:8000`:


![CatBench Normalized](/landing/catbench-normalized.png)

YouTube [videos are here](https://tanelpoder.com/posts/visualizing-embedding-vectors-as-heatmaps-videos/).

## Directory Structure

The app structure is deliberately very simple and flat. This is not a serious app, probably not efficient, secure or correct either. As I evolve it over time, I use this app for testing, measuring, learning more about high performance ML (and related) pipelines. I plan to include fancier stuff like Python GIL-avoidance, RDMA and GPUDirect and various different vector-search capable databases into this experiment at some point.

```
$ tree | grep -v jpg
.
├── app
│   ├── catvector.py
│   ├── heatmap.html
│   ├── heatmap.js
│   ├── index.html
│   └── style.css
├── data
│   ├── PetImages
│   │   ├── Cat
│   │   │   └── Thumbs.db
│   │   ├── CDLA-Permissive-2.0.pdf
│   │   ├── Dog
│   │   │   ├── dog_embeddings_500.tsv
│   │   │   └── Thumbs.db
│   │   ├── readme.txt
│   │   └── SameCat
│   └── README.md
├── embeddings
│   ├── cat_embeddings_small.tsv
│   ├── dog_embeddings_small.tsv
│   ├── plane_embeddings_small.tsv
│   └── samecat_11696.tsv
├── landing
│   ├── catbench-normalized.png
│   └── cat-dog-plane-embeddings-heatmap-annotated.png
├── LICENSE
├── README.md
├── requirements-imageproc.txt
├── requirements.txt
└── scripts
    ├── generate_embeddings.py
    └── plotemb.py

10 directories, 25383 files
```

The `data/PetImages` directory is the Kaggle Cat/Dog dataset (total 25k images) originally released by Microsoft:

* https://www.microsoft.com/en-us/download/details.aspx?id=54765

You don't need to separately download this file as it's already included in this repo (as permitted by Microsoft's CDLA license).
