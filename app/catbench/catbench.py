import os
import time
import textwrap
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from math import ceil

import psycopg2
from psycopg2 import pool
from flask import Flask, render_template, request, send_from_directory, abort, jsonify

app = Flask(__name__)

# Configuration
PG_DB   = "tanel"
PG_USER = "tanel"
PG_PASS = "tanel"
PG_HOST = "localhost"
PG_PORT = "5432"

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_DIR  = os.path.join(APP_DIR, 'data', 'PetImages')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ITEMS_PER_PAGE = 15

# Global variables for storing monitoring data
monitoring_data = {
    'last_sample_time': None,
    'previous_samples': {},
    'history': defaultdict(lambda: deque(maxlen=120))  # Keep last 120 samples (10 minutes at 5s interval)
}

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

# Monitoring routes and functions
@app.route('/monitoring')
def monitoring_page():
    return render_template('monitoring.html')

# API endpoint to get the latest monitoring data
@app.route('/api/monitoring_data')
def get_monitoring_data():
    interval = request.args.get('interval', 5, type=int)
    time_range = request.args.get('time_range', 5, type=int)
    fetch_latest_monitoring_data(interval)

    # Calculate max samples based on time range (in minutes) and interval
    max_samples = (time_range * 60) // interval

    # Prepare data for the frontend, limiting to the requested time range
    result = {
        'top_queries': list(monitoring_data['history']['top_queries'])[-max_samples:] if 'top_queries' in monitoring_data['history'] else [],
        'system_metrics': {}
    }

    # Add system metrics with the same time range limit
    for metric in ['cpu', 'memory', 'io', 'connections', 'query_efficiency']:
        if metric in monitoring_data['history']:
            result['system_metrics'][metric] = list(monitoring_data['history'][metric])[-max_samples:]
        else:
            result['system_metrics'][metric] = []

    return jsonify(result)

def clean_numeric_data(value):
    """Convert the value to a float if it's numeric, or return 0."""
    if value is None:
        return 0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0

def calculate_safe_delta(current, previous, time_delta):
    """Safely calculate rate (delta / time) with appropriate bounds checking."""
    if current is None or previous is None or time_delta <= 0:
        return 0

    try:
        current_val = float(current)
        previous_val = float(previous)

        # If the counter has been reset or gone backwards, just return 0
        if current_val < previous_val:
            return 0

        return (current_val - previous_val) / time_delta
    except (ValueError, TypeError):
        return 0

def fetch_latest_monitoring_data(interval=5):
    """
    Fetch the latest monitoring data from PostgreSQL and calculate deltas from the previous sample.
    """
    current_time = time.time()

    # If we haven't sampled before or it's time for a new sample
    if monitoring_data['last_sample_time'] is None or (current_time - monitoring_data['last_sample_time']) >= interval:
        conn = get_db()
        try:
            # Get top queries from pg_stat_statements
            top_queries = fetch_top_queries(conn)

            # Get system-wide metrics
            system_metrics = fetch_system_metrics(conn)

            # Calculate deltas if we have previous samples
            if monitoring_data['previous_samples']:
                time_delta = current_time - monitoring_data['last_sample_time']

                # Calculate deltas for top queries
                for query in top_queries:
                    query_id = query['queryid']
                    if query_id in monitoring_data['previous_samples']['top_queries']:
                        prev = monitoring_data['previous_samples']['top_queries'][query_id]

                        # Calculate calls per second
                        calls_delta = clean_numeric_data(query['calls']) - clean_numeric_data(prev['calls'])
                        query['calls_per_sec'] = calculate_safe_delta(query['calls'], prev['calls'], time_delta)

                        # Only calculate other metrics if there were actual calls
                        if calls_delta > 0:
                            query['total_exec_time_delta'] = clean_numeric_data(query['total_exec_time']) - clean_numeric_data(prev['total_exec_time'])
                            query['avg_exec_time_delta'] = query['total_exec_time_delta'] / calls_delta

                            # Calculate block-related metrics
                            for metric in ['shared_blks_hit', 'shared_blks_read', 'temp_blks_read', 'temp_blks_written']:
                                query[f'{metric}_per_sec'] = calculate_safe_delta(query[metric], prev[metric], time_delta)
                        else:
                            # No calls in this period, set rates to zero
                            query['total_exec_time_delta'] = 0
                            query['avg_exec_time_delta'] = 0
                            for metric in ['shared_blks_hit', 'shared_blks_read', 'temp_blks_read', 'temp_blks_written']:
                                query[f'{metric}_per_sec'] = 0

                # Calculate deltas for system metrics
                for metric in system_metrics:
                    if metric in monitoring_data['previous_samples']['system_metrics']:
                        prev = monitoring_data['previous_samples']['system_metrics'][metric]

                        if isinstance(system_metrics[metric], dict) and isinstance(prev, dict):
                            # Create a new dictionary for the per-second rates
                            for key in list(system_metrics[metric].keys()):
                                # Skip keys that already have _per_sec suffix
                                if key.endswith('_per_sec'):
                                    continue

                                # Skip non-numeric values
                                if not isinstance(system_metrics[metric][key], (int, float)):
                                    continue

                                if key in prev and isinstance(prev[key], (int, float)):
                                    # Calculate rate as (current - previous) / time_delta
                                    rate = calculate_safe_delta(system_metrics[metric][key], prev[key], time_delta)
                                    system_metrics[metric][f"{key}_per_sec"] = rate

            # Store current samples as previous for next run
            monitoring_data['previous_samples'] = {
                'top_queries': {query['queryid']: query for query in top_queries},
                'system_metrics': system_metrics
            }

            # Store current time
            monitoring_data['last_sample_time'] = current_time

            # Add to history
            timestamp = datetime.now().strftime('%H:%M:%S')

            # Transform top_queries into a format suitable for charts
            valid_queries = [q for q in top_queries if 'calls_per_sec' in q]
            top_queries_data = {
                'timestamp': timestamp,
                'queries': sorted(
                    valid_queries,
                    key=lambda x: x.get('total_exec_time_delta', 0),
                    reverse=True
                )[:10]  # Top 10 queries by execution time
            }
            monitoring_data['history']['top_queries'].append(top_queries_data)

            # Add system metrics to history
            for metric, value in system_metrics.items():
                # Create a data point for this metric
                data_point = {'timestamp': timestamp}

                if isinstance(value, dict):
                    # Extract rate values (those ending with _per_sec) if they exist
                    rate_values = {k: v for k, v in value.items() if k.endswith('_per_sec') and isinstance(v, (int, float))}
                    if rate_values:
                        # If we have rate values, add them to the data point
                        data_point.update(rate_values)
                    else:
                        # Otherwise, just include the original values
                        data_point.update({k: v for k, v in value.items() if isinstance(v, (int, float))})
                elif isinstance(value, (int, float)):
                    # If the value is a simple number, add it directly
                    data_point['value'] = value
                else:
                    # Skip this metric if it's not a number or dictionary
                    continue

                # Add the data point to the history
                monitoring_data['history'][metric].append(data_point)

        finally:
            release_db(conn)

def fetch_top_queries(conn):
    """
    Fetch top queries from pg_stat_statements ordered by total execution time.
    """
    top_queries = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    queryid,
                    query,
                    calls,
                    total_exec_time,
                    min_exec_time,
                    max_exec_time,
                    mean_exec_time,
                    stddev_exec_time,
                    rows / calls AS rows,
                    shared_blks_hit,
                    shared_blks_read,
                    shared_blks_dirtied,
                    shared_blks_written,
                    local_blks_hit,
                    local_blks_read,
                    local_blks_dirtied,
                    local_blks_written,
                    temp_blks_read,
                    temp_blks_written,
                    blk_read_time,
                    blk_write_time
                FROM pg_stat_statements
                WHERE queryid IS NOT NULL
                ORDER BY total_exec_time DESC
                LIMIT 20
            """)
            for row in cur.fetchall():
                # Convert row to dictionary
                query_data = {
                    'queryid': row[0],
                    'query': row[1],
                    'calls': row[2],
                    'total_exec_time': row[3],
                    'min_exec_time': row[4],
                    'max_exec_time': row[5],
                    'mean_exec_time': row[6],
                    'stddev_exec_time': row[7],
                    'rows': row[8],
                    'shared_blks_hit': row[9],
                    'shared_blks_read': row[10],
                    'shared_blks_dirtied': row[11],
                    'shared_blks_written': row[12],
                    'local_blks_hit': row[13],
                    'local_blks_read': row[14],
                    'local_blks_dirtied': row[15],
                    'local_blks_written': row[16],
                    'temp_blks_read': row[17],
                    'temp_blks_written': row[18],
                    'blk_read_time': row[19],
                    'blk_write_time': row[20]
                }
                top_queries.append(query_data)
    except Exception as e:
        print(f"Error fetching top queries: {e}")
        # If pg_stat_statements is not available, return empty list
        return []

    return top_queries

def fetch_system_metrics(conn):
    """
    Fetch system-wide PostgreSQL metrics.
    """
    system_metrics = {}
    try:
        with conn.cursor() as cur:
            # Database statistics
            cur.execute("""
                SELECT
                    sum(numbackends) as connections,
                    sum(xact_commit) as commits,
                    sum(xact_rollback) as rollbacks,
                    sum(blks_read) as disk_reads,
                    sum(blks_hit) as buffer_hits,
                    sum(tup_returned) as rows_returned,
                    sum(tup_fetched) as rows_fetched,
                    sum(tup_inserted) as rows_inserted,
                    sum(tup_updated) as rows_updated,
                    sum(tup_deleted) as rows_deleted
                FROM pg_stat_database
            """)
            row = cur.fetchone()
            system_metrics['database'] = {
                'connections': row[0] or 0,
                'commits': row[1] or 0,
                'rollbacks': row[2] or 0,
                'disk_reads': row[3] or 0,
                'buffer_hits': row[4] or 0,
                'rows_returned': row[5] or 0,
                'rows_fetched': row[6] or 0,
                'rows_inserted': row[7] or 0,
                'rows_updated': row[8] or 0,
                'rows_deleted': row[9] or 0
            }

            # Buffer statistics
            try:
                cur.execute("""
                    SELECT
                        buffers_checkpoint,
                        buffers_clean,
                        buffers_backend,
                        buffers_backend_fsync,
                        buffers_alloc
                    FROM pg_stat_bgwriter
                """)
                row = cur.fetchone()
                if row:
                    system_metrics['bgwriter'] = {
                        'buffers_checkpoint': row[0] or 0,
                        'buffers_clean': row[1] or 0,
                        'buffers_backend': row[2] or 0,
                        'buffers_backend_fsync': row[3] or 0,
                        'buffers_alloc': row[4] or 0
                    }
                else:
                    system_metrics['bgwriter'] = {
                        'buffers_checkpoint': 0,
                        'buffers_clean': 0,
                        'buffers_backend': 0,
                        'buffers_backend_fsync': 0,
                        'buffers_alloc': 0
                    }
            except Exception as e:
                print(f"Error fetching bgwriter statistics: {e}")
                system_metrics['bgwriter'] = {
                    'buffers_checkpoint': 0,
                    'buffers_clean': 0,
                    'buffers_backend': 0,
                    'buffers_backend_fsync': 0,
                    'buffers_alloc': 0
                }

            # Connection statistics
            try:
                cur.execute("""
                    SELECT count(*) FROM pg_stat_activity
                """)
                system_metrics['connections'] = cur.fetchone()[0] or 0
            except Exception as e:
                print(f"Error fetching connection statistics: {e}")
                system_metrics['connections'] = 0

            # Memory usage
            try:
                cur.execute("""
                    SELECT
                        COALESCE(sum(pg_total_relation_size(c.oid)), 0) as total_table_size,
                        COALESCE(sum(pg_indexes_size(c.oid)), 0) as total_index_size,
                        COALESCE(sum(pg_total_relation_size(c.oid) - pg_relation_size(c.oid)), 0) as total_external_size
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                    AND c.relkind IN ('r', 't')
                """)
                row = cur.fetchone()
                system_metrics['memory'] = {
                    'total_table_size': row[0] or 0,
                    'total_index_size': row[1] or 0,
                    'total_external_size': row[2] or 0
                }
            except Exception as e:
                print(f"Error fetching memory statistics: {e}")
                system_metrics['memory'] = {
                    'total_table_size': 0,
                    'total_index_size': 0,
                    'total_external_size': 0
                }

            # I/O statistics
            try:
                cur.execute("""
                    SELECT
                        COALESCE(sum(heap_blks_read), 0) as heap_read,
                        COALESCE(sum(heap_blks_hit), 0) as heap_hit,
                        COALESCE(sum(idx_blks_read), 0) as idx_read,
                        COALESCE(sum(idx_blks_hit), 0) as idx_hit,
                        COALESCE(sum(toast_blks_read), 0) as toast_read,
                        COALESCE(sum(toast_blks_hit), 0) as toast_hit,
                        COALESCE(sum(tidx_blks_read), 0) as tidx_read,
                        COALESCE(sum(tidx_blks_hit), 0) as tidx_hit
                    FROM pg_statio_all_tables
                """)
                row = cur.fetchone()
                if row:
                    system_metrics['io'] = {
                        'heap_read': row[0] or 0,
                        'heap_hit': row[1] or 0,
                        'idx_read': row[2] or 0,
                        'idx_hit': row[3] or 0,
                        'toast_read': row[4] or 0,
                        'toast_hit': row[5] or 0,
                        'tidx_read': row[6] or 0,
                        'tidx_hit': row[7] or 0
                    }
                else:
                    system_metrics['io'] = {
                        'heap_read': 0,
                        'heap_hit': 0,
                        'idx_read': 0,
                        'idx_hit': 0,
                        'toast_read': 0,
                        'toast_hit': 0,
                        'tidx_read': 0,
                        'tidx_hit': 0
                    }
            except Exception as e:
                print(f"Error fetching I/O statistics: {e}")
                system_metrics['io'] = {
                    'heap_read': 0,
                    'heap_hit': 0,
                    'idx_read': 0,
                    'idx_hit': 0,
                    'toast_read': 0,
                    'toast_hit': 0,
                    'tidx_read': 0,
                    'tidx_hit': 0
                }

            # Query efficiency statistics
            try:
                cur.execute("""
                    SELECT
                        COALESCE(sum(calls), 0) as total_calls,
                        COALESCE(sum(total_exec_time), 0) as total_time,
                        COALESCE(sum(rows), 0) as total_rows
                    FROM pg_stat_statements
                """)
                row = cur.fetchone()
                system_metrics['query_efficiency'] = {
                    'total_calls': row[0] or 0,
                    'total_time': row[1] or 0,
                    'total_rows': row[2] or 0
                }
            except Exception as e:
                print(f"Error fetching query efficiency statistics: {e}")
                system_metrics['query_efficiency'] = {
                    'total_calls': 0,
                    'total_time': 0,
                    'total_rows': 0
                }

    except Exception as e:
        print(f"Error fetching system metrics: {e}")

    return system_metrics

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
