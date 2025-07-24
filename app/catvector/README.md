### Static CatVector app that doesn't require a database

The static embedding vector visualization blog entry and demo videos are here:

* https://tanelpoder.com/posts/visualizing-embedding-vectors-as-heatmaps/
* https://tanelpoder.com/posts/visualizing-embedding-vectors-as-heatmaps-videos/

![Cats Dogs Planes](/landing/cat-dog-plane-embeddings-heatmap-annotated.webp)


## CatVector install instructions

```
pip install -r requirements-catvector.txt

# if you want airplane images 
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
tar xf fgvc-aircraft-2013b.tar.gz
cd ..

# run the app
cd app/catvector
python catvector.py
```
Then go to `hostname:8000`:

![CatBench Normalized](/landing/catbench-normalized.png)

