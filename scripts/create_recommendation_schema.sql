-- customize this based on CPUs and free memory of your server
SET maintenance_work_mem = '32GB';
SET max_parallel_workers = 24;
SET max_parallel_maintenance_workers = 16;


-- vectors generated using pytorch/ViT embedding model from the original Kaggle cats & dogs image set

CREATE TABLE customer_fingerprints (
    fingerprint_id   SERIAL PRIMARY KEY
  , c_id             INT NOT NULL
  , w_id             INT NOT NULL
  , d_id             INT NOT NULL
  , cust_type        VARCHAR(10) -- cat or dog
  , file_name        VARCHAR(100)
  , embedding        VECTOR(1000)
);

-- we have 30k customers: 3000 distinct customer_ids * 10 districts * 1 warehouse,
-- thanks to how HammerDB/TPCC schema and data generation works
-- we have 25k pet photos (12500 cats + 12500 dogs), so most of the customers will 
-- have a pet photo fingerprint linked to them, using the hacky-looking SQL below:

INSERT INTO customer_fingerprints (c_id, w_id, d_id, cust_type, file_name, embedding)
SELECT MOD(id,1500)+1,    1, MOD(id,10)+1, 'cat', file_name, embedding FROM cats
UNION ALL
SELECT MOD(id,1500)+1501, 1, MOD(id,10)+1, 'dog', file_name, embedding FROM dogs;

-- create indexes
CREATE INDEX idx_cust_fp ON customer_fingerprints (c_id, w_id, d_id);
CREATE INDEX hnsw_cust_fp_embedding ON customer_fingerprints USING hnsw (embedding vector_l2_ops);
ANALYZE customer_fingerprints;


-- daily batch job for recomputing individual customers top purchases
-- since most customers don't buy the same product many times in this
-- dataset, I changed the ranking to use the "total_spent" number

CREATE TABLE customer_top_items AS
WITH customer_items AS (
    SELECT 
        c.c_id
      , c.c_d_id
      , c.c_w_id
      , i.i_id
      , i.i_name
      , COUNT(*) as purchase_count
      , COUNT(*) * i.i_price as total_spent
      , ROW_NUMBER() OVER (PARTITION BY c.c_id, c.c_d_id, c.c_w_id ORDER BY COUNT(*) * i.i_price DESC) as rank
    FROM customer c
    INNER JOIN district d 
       ON c.c_d_id = d.d_id 
      AND c.c_w_id = d.d_w_id
    INNER JOIN warehouse w 
       ON c.c_w_id = w.w_id
    INNER JOIN orders o 
       ON o.o_c_id = c.c_id 
      AND o.o_w_id = c.c_w_id 
      AND o.o_d_id = c.c_d_id
    INNER JOIN order_line ol 
       ON ol.ol_o_id = o.o_id 
      AND ol.ol_w_id = o.o_w_id 
      AND ol.ol_d_id = o.o_d_id
    INNER JOIN stock s 
       ON ol.ol_w_id = s.s_w_id 
      AND ol.ol_i_id = s.s_i_id
    INNER JOIN item i 
      ON s.s_i_id = i.i_id
    GROUP BY
        c.c_id
      , c_d_id
      , c_w_id
      , i.i_id
      , i.i_name
)
SELECT 
    c_id
  , c_w_id
  , c_d_id
  , i_id
  , i_name
  , purchase_count
  , total_spent
  , rank
FROM customer_items
WHERE rank <= 5;


CREATE INDEX idx_cust_top_items ON customer_top_items (c_id, c_d_id, c_w_id, rank);
ANALYZE customer_top_items;

