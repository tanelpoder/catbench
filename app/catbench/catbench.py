import os
import textwrap
from pathlib import Path
from math import ceil

import psycopg2
from psycopg2 import pool
from flask import Flask, render_template, request, send_from_directory, abort, jsonify

import monitoring

app = Flask(__name__)

# Configuration (local socket using your OS user for username and dbname by default)
PG_DB   = None    # DBname defaults to your OS username, or replace with your dbname
PG_USER = None    # Change to custom username if you don't want to use your local OS username
PG_PASS = None    # Change to your password if your pg_hba.conf doesn't trust local connections
PG_HOST = None    # change to your DB instance's hostname if using a remote DB
PG_PORT = '5432'

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR  = os.path.join(APP_DIR, 'data', 'PetImages')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ITEMS_PER_PAGE = 15

# Database connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 10,
    database=PG_DB,
    user=PG_USER,
    password=PG_PASS,
    host=PG_HOST,
    port=PG_PORT
)

def get_db():
    return db_pool.getconn()

def release_db(conn):
    db_pool.putconn(conn)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def landing_page():
    return render_template('landing.html')

def get_gallery_images(page, animal):
    animal_dir = os.path.join(IMAGE_DIR, animal.capitalize())

    # list the gallery images by their filename numeric part for now
    images = sorted(
        [f for f in os.listdir(animal_dir) if allowed_file(f)],
         key=lambda f: int(os.path.splitext(f)[0])
    )


    total_images = len(images)
    total_pages = ceil(total_images / ITEMS_PER_PAGE)

    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    paginated_images = images[start_idx:end_idx]

    return paginated_images, total_pages

@app.route('/identify/cat')
def identify_cat_gallery():
    page = request.args.get('page', 1, type=int)
    paginated_images, total_pages = get_gallery_images(page, 'cat')
    return render_template('gallery.html',
                         images=paginated_images,
                         page=page,
                         total_pages=total_pages,
                         mode='identify',
                         animal='cat',
                         title='Cat-fraud Detection')

@app.route('/similar/cat')
def similar_cat_gallery():
    page = request.args.get('page', 1, type=int)
    paginated_images, total_pages = get_gallery_images(page, 'cat')
    return render_template('gallery.html',
                         images=paginated_images,
                         page=page,
                         total_pages=total_pages,
                         mode='similar',
                         animal='cat',
                         title='Cat Recommendation Engine')

@app.route('/identify/dog')
def identify_dog_gallery():
    page = request.args.get('page', 1, type=int)
    paginated_images, total_pages = get_gallery_images(page, 'dog')
    return render_template('gallery.html',
                         images=paginated_images,
                         page=page,
                         total_pages=total_pages,
                         mode='identify',
                         animal='dog',
                         title='Dog-fraud Detection')

@app.route('/similar/dog')
def similar_dog_gallery():
    page = request.args.get('page', 1, type=int)
    paginated_images, total_pages = get_gallery_images(page, 'dog')
    return render_template('gallery.html',
                         images=paginated_images,
                         page=page,
                         total_pages=total_pages,
                         mode='similar',
                         animal='dog',
                         title='Dog Recommendation Engine')

@app.route('/crossspecies/cat')
def catlike_dogs_gallery():
    page = request.args.get('page', 1, type=int)
    paginated_images, total_pages = get_gallery_images(page, 'cat')
    return render_template('gallery.html',
                         images=paginated_images,
                         page=page,
                         total_pages=total_pages,
                         mode='crossspecies',
                         animal='cat',
                         title='Find Cat Lookalikes')

@app.route('/crossspecies/dog')
def doglike_cats_gallery():
    page = request.args.get('page', 1, type=int)
    paginated_images, total_pages = get_gallery_images(page, 'dog')
    return render_template('gallery.html',
                         images=paginated_images,
                         page=page,
                         total_pages=total_pages,
                         mode='crossspecies',
                         animal='dog',
                         title='Find Dog Lookalikes')

@app.route('/<animal>/<mode>/<image_name>')
def pet_details(animal, mode, image_name):
    if not allowed_file(image_name) or animal not in ['cat', 'dog']:
        abort(404)

    # Define table names based on mode and animal
    if mode == 'identify':               # cat-fraud detection
        source_table  = f"{animal}s"     # cats or dogs
        compare_table = f"many{animal}s" # manycats/dogs (0..359 degree rotated variants of each)
    elif mode == 'crossspecies':
        source_table = f"{animal}s"      # source species table (origcats or origdogs)
        compare_table = "dogs" if animal == "cat" else "cats"
    else:                                # similar mode
        source_table = f"{animal}s"      # cats or dogs
        compare_table = source_table     # same table for comparison

    conn = get_db()
    query_plans = []
    try:
        with conn.cursor() as cur:
            # First query: Get exec plan and stats for retrieving image files (precomputed) embedding
            embedding_query = f"SELECT embedding FROM {source_table} WHERE file_name = %s LIMIT 1"
            cur.execute(f"EXPLAIN (ANALYZE,BUFFERS) {embedding_query}", (f"{image_name}",))
            query_plans.append({
                'query': embedding_query,
                'plan': '\n'.join([row[0] for row in cur.fetchall()])
            })

            # Execute the same query to get the data
            cur.execute(embedding_query, (f"{image_name}",))
            embedding = cur.fetchone()[0]

            # Second query: Get similar images
            if mode == 'similar':
                similarity_query = textwrap.dedent(f"""\
                                     SELECT file_name, embedding <-> %s::vector AS distance
                                     FROM {compare_table}
                                     WHERE file_name != %s
                                     ORDER BY embedding <-> %s::vector
                                     LIMIT 20""")
                params = (embedding, image_name, embedding)
            else:
                similarity_query = textwrap.dedent(f"""\
                                     SELECT file_name, embedding <-> %s::vector AS distance
                                     FROM {compare_table}
                                     ORDER BY embedding <-> %s::vector
                                     LIMIT 20""")
                params = (embedding, embedding)

            # Get execution plan for similarity query
            cur.execute(f"EXPLAIN (ANALYZE,BUFFERS) {similarity_query}", params)
            query_plans.append({
                'query': similarity_query,
                'plan': '\n'.join([row[0] for row in cur.fetchall()])
            })

            # Execute actual similarity query
            cur.execute(similarity_query, params)
            results = cur.fetchall()

            # Only execute purchase analysis for cat recommendations for now
            purchase_analysis = None
            if mode == 'similar':
                purchase_query = textwrap.dedent("""\
                    WITH similar_customers AS (
                        SELECT c_id, w_id, d_id, cust_type
                        FROM customer_fingerprints
                        ORDER BY embedding <-> %s::vector
                        LIMIT 20
                    )
                    SELECT
                        COUNT(*),
                        t.i_id,
                        t.i_name,
                        SUM(t.purchase_count) total_purchase_count,
                        SUM(t.total_spent) total_spent,
                        sc.cust_type
                    FROM
                        similar_customers sc,
                        customer_top_items t
                    WHERE
                        sc.c_id = t.c_id
                    AND sc.w_id = t.c_w_id
                    AND sc.d_id = t.c_d_id
                    GROUP BY
                        t.i_id,
                        t.i_name,
                        sc.cust_type
                    ORDER BY
                        total_spent DESC
                    LIMIT 10
                """)

                try:
                    # Get execution plan and metrics for purchase analysis
                    cur.execute(f"EXPLAIN (ANALYZE,BUFFERS) {purchase_query}", (embedding,))
                    query_plans.append({
                        'query': purchase_query,
                        'plan': '\n'.join([row[0] for row in cur.fetchall()])
                    })

                    # Re-run the same query to get its output data
                    cur.execute(purchase_query, (embedding,))
                    purchase_analysis = cur.fetchall()

                except Exception as e:
                    print(f"ERROR: Failed to execute purchase analysis query: {e}")
                    purchase_analysis = None

    finally:
        release_db(conn)

    similar_images = [{"filename": result[0], "distance": f"{result[1]:.4f}"} for result in results]

    return render_template('pet_details.html',
                         main_image=image_name,
                         similar_images=similar_images,
                         mode=mode,
                         animal=animal,
                         purchase_analysis=purchase_analysis,
                         query_plans=query_plans)


# compare_image is the "incoming" image that we have to match against the
# source_image that is the "known" customer in the CRM system (Cat Relationsip Management system)
@app.route('/<animal>/identify/reverse/<source_image>/<compare_image>')
def reverse_lookup(animal, source_image, compare_image):
    if not allowed_file(source_image) or not allowed_file(compare_image) or animal not in ['cat', 'dog']:
        abort(404)

    conn = get_db()
    query_plans = []
    try:
        with conn.cursor() as cur:
            # Get embeddings for compare_image now to see if it actually matches
            # the original "known" pet in the CRM closely enough
            compare_query = textwrap.dedent(f"""\
                             SELECT embedding
                             FROM {animal}s
                             WHERE file_name = %s
                             LIMIT 1""")

            # Get incoming pic execution plan and stats first
            cur.execute(f"EXPLAIN (ANALYZE,BUFFERS) {compare_query}", (compare_image,))
            query_plans.append({
                'query': compare_query,
                'plan': '\n'.join([row[0] for row in cur.fetchall()])
            })

            # Re-run the incoming pic query to get the output
            cur.execute(compare_query, (compare_image,))
            compare_embedding = cur.fetchone()[0]

            # Now query for closest match from existing known cat-customers using vector similarity
            source_query = textwrap.dedent(f"""\
                              SELECT file_name, embedding, embedding <-> %s::vector AS distance
                              FROM {animal}s
                              WHERE file_name != %s
                              ORDER BY embedding <-> %s::vector
                              LIMIT 1""")

            # Get execution plan for compare image query
            cur.execute(f"EXPLAIN (ANALYZE,BUFFERS) {source_query}", (compare_embedding, compare_image, compare_embedding))
            query_plans.append({
                'query': source_query,
                'plan': '\n'.join([row[0] for row in cur.fetchall()])
            })

            # Execute actual compare query
            cur.execute(source_query, (compare_embedding, compare_image, compare_embedding))
            closest_match = cur.fetchone()
            closest_file_name = closest_match[0]
            source_embedding = closest_match[1]
            vector_distance = closest_match[2]

            # Add information about whether this was a symmetric match (ignoring the image rotation prefix)
            is_symmetric_match = closest_file_name[4:] == source_image

            # Calculate vector distance
            distance_query = "SELECT %s::vector <-> %s::vector AS distance"

            # Get execution plan for distance calculation
            cur.execute(f"EXPLAIN (ANALYZE,BUFFERS) {distance_query}",
                       (source_embedding, compare_embedding))
            query_plans.append({
                'query': distance_query,
                'plan': '\n'.join([row[0] for row in cur.fetchall()])
            })

            # Execute actual distance calculation
            cur.execute(distance_query, (compare_embedding, source_embedding))
            distance = cur.fetchone()[0]

    finally:
        release_db(conn)

    return render_template('reverse_lookup.html',
                         source_image=source_image,
                         compare_image=compare_image,
                         distance=f"{vector_distance:.4f}",
                         animal=animal,
                         query_plans=query_plans,
                         is_symmetric_match=is_symmetric_match,
                         closest_file_name=closest_file_name)



@app.route('/images/<animal>/<filename>')
def image_file(animal, filename):
    if animal not in ['cat', 'dog']:
        abort(404)

    # if having millions of rotated files in respective subdirs
    if '_' in filename:
        subfolder = filename.split('_')[0]
        image_path = os.path.join(IMAGE_DIR, animal.capitalize(), subfolder)
    else:
        image_path = os.path.join(IMAGE_DIR, animal.capitalize())

    return send_from_directory(image_path, filename)

# Monitoring routes
@app.route('/monitoring')
def monitoring_page():
    return render_template('monitoring.html')

# API endpoint to get the latest monitoring data
@app.route('/api/monitoring_data')
def get_monitoring_data():
    interval = request.args.get('interval', 5, type=int)
    time_range = request.args.get('time_range', 5, type=int)

    # Fetch latest monitoring data using the monitoring module
    monitoring.fetch_latest_monitoring_data(get_db, release_db, interval)

    # Calculate max samples based on time range (in minutes) and interval
    max_samples = (time_range * 60) // interval

    # Get monitoring data from the module
    result = monitoring.get_monitoring_data(time_range, max_samples)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
