#CatBench Vector Search Playground
_Cat Benchmarking at Scale, finally!_

There are two separate Python apps in the [app](https://github.com/tanelpoder/catbench/tree/main/app) directory:

* **CatVector** - a simple static vector heatmap visualization app (no database)
* **CatBench** - a simple Python/Flask application using Postgres+pgvector similarity search queries (and joins to a regular TPCC schema)

[Go to installation steps below](#installation-steps)


## CatBench

You can test this app out yourself, installation steps are below.

Here are a few screenshots of the similarity search and recommendation engine app (for cats!) in action:

![Cat similarity search output](/landing/rec-engine-1.webp)
![Cat similarity search query](/landing/rec-engine-2.webp)
![Cat recommendation engine output](/landing/rec-engine-3.webp)
![Cat recommendation engine query plan](/landing/rec-engine-4.webp)
![Cat recommendation engine query plan](/landing/catbench-monitoring-1.webp)

## Installation Steps

25000 cat/dog images are included in this repository. If you want to download aircraft images too, use the `wget` or `curl` command below. I have tested this on RHEL9 and Ubuntu 24.04 so far. You need to have python and pip installed in your OS for this. For installing Python packages locally with `pip`, you probably want to use a Python virtual environment (venv).

### Interactive CatBench application that requires a Postgres database and loading data

Make sure that you have a Postgres database (with pgvector extension) running and accessible and change the `psql` commands below to include your username/password if you are not using a default local connection:

In the `catbench` repo root directory, run this to generate embedding vectors from the 25000 pet images (this uses PyTorch which automatically runs on CPUs if you don't have a GPU available).

```
git clone https://github.com/tanelpoder/catbench
cd catbench

pip install -r requirements-catbench.txt
```

**NB!** You may need to install Postgres and the PgVector extension _and_ the `python3-psycopg2` package using your OS package manager first, if `pip` doesn't successfully install `psycopg2` on your Linux distro.

The next step generates vector embeddings for the 25000 pet photos included in this repository (using GPU's if cuda/NVIDIA GPUs are available, otherwise CPUs.

Process the 25000 pet photos and generate their embeddings for loading into postgres:

```
python app/catbench/scripts/generate_embeddings.py data/PetImages/Cat embeddings/cats.tsv
python app/catbench/scripts/generate_embeddings.py data/PetImages/Dog embeddings/dogs.tsv
```

This may take a while. Then load the vectors and other OLTP data into the database:

```
gunzip  app/catbench/scripts/create_tpcc_tables.sql.gz
psql -f app/catbench/scripts/create_tpcc_tables.sql 
psql -f app/catbench/scripts/create_catbench_tables.sql 
psql -f app/catbench/scripts/create_recommendation_schema.sql 
```

If you're using a local Postgres instance that allows logging in as `tpcc` user without a password, no action needed. Otherwise open the `catbench.py` file to change your Postgres user/pass settings if you are not using a default local connection. And then run the app:

```
cd app/catbench

python3 catbench.py
```

You can now go to `hostname:5000` and browse around:

![CatBench app frontpage](/landing/catbench-frontpage.webp)

### Stress test

* Check the [app/catbench/scripts/](https://github.com/tanelpoder/catbench/tree/main/app/catbench/scripts) directory and run `cat_loop.sh` or `cat_loop_wit_recall.sh` scripts in there (the same for dogs). These shell scripts call similarly named `.sql` scripts under the hood, look inside them to see how they work. You can use similar patterns to construct your own stress test queries.
* You currently need to change the "tpcc" to your database name (if you're not using "tpcc").
* You can uncomment more `psql` lines to increase concurrency (and hit CTRL+C in terminal to cancel/kill all currently running `psql` loops`

### Other

The `data/PetImages` directory is the Kaggle Cat/Dog dataset (total 25k images) originally released by Microsoft:

* https://www.microsoft.com/en-us/download/details.aspx?id=54765

You don't need to separately download this file as it's already included in this repo (as permitted by Microsoft's CDLA license).
