import time
from datetime import datetime
from collections import defaultdict, deque

# Global variables for storing monitoring data
monitoring_data = {
    'last_sample_time': None,
    'previous_samples': {},
    'history': defaultdict(lambda: deque(maxlen=120))  # Keep last 120 samples (10 minutes at 5s interval)
}


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


def fetch_latest_monitoring_data(get_db_func, release_db_func, interval=5):
    """
    Fetch the latest monitoring data from PostgreSQL and calculate deltas from the previous sample.
    """
    current_time = time.time()

    # If we haven't sampled before or it's time for a new sample
    if monitoring_data['last_sample_time'] is None or (current_time - monitoring_data['last_sample_time']) >= interval:
        conn = get_db_func()
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
            release_db_func(conn)


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


def get_monitoring_data(time_range, max_samples):
    """
    Get monitoring data for the frontend, limiting to the requested time range.
    """
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

    return result
