# CatBench Vector Search Playground
_Cat Benchmarking at Scale, finally!_

There are two separate Python apps in this repo:

* **CatVector** - a simple static embedding vector visualization app that shows a heatmap of individual vector/array values from various cat, dog and plane photos. Everything is precomputed and stored in text files, so you don't need PyTorch, GPUs or even a database
* **CatBench** - a simple Python/Flask application using Postgres+pgvector similarity search queries (and joins to a regular TPCC schema) for playing around with vector similarity search based use cases

## CatBench

You can already test this app out yourself, installation steps are below.

Here are a few screenshots of the similarity search and recommendation engine app (for cats!) in action:

![Cat similarity search output](/landing/rec-engine-1.webp)
![Cat similarity search query](/landing/rec-engine-2.webp)
![Cat recommendation engine output](/landing/rec-engine-3.webp)
![Cat recommendation engine query plan](/landing/rec-engine-4.webp)

## CatVector 

The static embedding vector visualization blog entry and demo videos are here:

* https://tanelpoder.com/posts/visualizing-embedding-vectors-as-heatmaps/
* https://tanelpoder.com/posts/visualizing-embedding-vectors-as-heatmaps-videos/

![Cats Dogs Planes](/landing/cat-dog-plane-embeddings-heatmap-annotated.webp)

## Installation Steps

25000 cat/dog images are included in this repository. If you want to download aircraft images too, use the `wget` command below. I have tested this on RHEL9 and Ubuntu 24.04 so far. You need to have python and pip installed in your OS for this. For installing Python packages locally with `pip`, you probably want to use a Python virtual environment (venv).

Download and set up CatBench:

### Static CatVector app that doesn't require a database

```
git clone https://github.com/tanelpoder/catbench
cd catbench

pip install -r requirements-catvector.txt

# if you want airplane images
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar xf fgvc-aircraft-2013b.tar.gz

# run the app
cd ../app/catvector
python catvector.py
```
Then go to `hostname:8000`:

![CatBench Normalized](/landing/catbench-normalized.png)

### Interactive CatBench application that requires a Postgres database and loading data

Make sure that you have a Postgres database running and accessible and change the `psql` commands below to include your username/password if you are not using a default local connection:

In the `catbench` repo root directory, run this to generate embedding vectors from the 25000 pet images (this uses PyTorch which automatically runs on CPUs if you don't have a GPU available).

```
pip install -r requirements-catbench.txt
python scripts/generate_embeddings.py data/PetImages/Cat embeddings/cats.tsv
python scripts/generate_embeddings.py data/PetImages/Dog embeddings/dogs.tsv
```

Then load the vectors and other data into the database:

```
gunzip scripts/create_tpcc_tables.sql.gz
psql -f scripts/create_tpcc_tables.sql 
psql -f scripts/create_catbench_tables.sql 
psql -f scripts/create_recommendation_schema.sql 
```

Now go to the CatBench app directory:

```
cd app/catbench
```

Open the `catbench.py` file to change your Postgres user/pass settings if you are not using a default local connection. And then run the app:

```
python catbench.py
```

You can now go to `servername:5000` and browse around:

![CatBench app frontpage](/landing/catbench-frontpage.webp)

The `data/PetImages` directory is the Kaggle Cat/Dog dataset (total 25k images) originally released by Microsoft:

* https://www.microsoft.com/en-us/download/details.aspx?id=54765

You don't need to separately download this file as it's already included in this repo (as permitted by Microsoft's CDLA license).
