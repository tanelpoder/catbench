-- customize this based on CPUs and free memory of your server
SET maintenance_work_mem = '32GB';
SET max_parallel_workers = 24;
SET max_parallel_maintenance_workers = 16;

-- create tables
CREATE TABLE cats (id SERIAL PRIMARY KEY, file_name VARCHAR(100), embedding VECTOR(1000));
CREATE TABLE dogs (id SERIAL PRIMARY KEY, file_name VARCHAR(100), embedding VECTOR(1000));

-- load data (12500 cats and 12500 dogs)
\COPY CATS(file_name, embedding) FROM 'embeddings/cats.tsv'  WITH (FORMAT csv, DELIMITER E'\t', NULL '', HEADER false);
\COPY DOGS(file_name, embedding) FROM 'embeddings/dogs.tsv'  WITH (FORMAT csv, DELIMITER E'\t', NULL '', HEADER false);

-- create aditional indexes (including vector indexes) 
CREATE INDEX idx_cats_file_name ON cats (file_name);
CREATE INDEX hnsw_cats_embedding ON cats USING hnsw (embedding vector_l2_ops);

CREATE INDEX idx_dogs_file_name ON dogs (file_name);
CREATE INDEX hnsw_dogs_embedding ON dogs USING hnsw (embedding vector_l2_ops);

