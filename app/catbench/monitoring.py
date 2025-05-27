"""PostgreSQL monitoring module for CatBench."""
import time
from collections import defaultdict, deque
from datetime import datetime


# Global monitoring data storage
monitoring_data = {
    'last_sample_time': None,
    'previous_samples': {},
    'history': defaultdict(lambda: deque(maxlen=120))  # 10 minutes at 5s interval
}


def safe_float(value, default=0):
    """Convert to float safely."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def calculate_rate(current, previous, time_delta):
    """Calculate rate per second."""
    if time_delta <= 0:
        return 0
    curr = safe_float(current)
    prev = safe_float(previous)
    return max(0, (curr - prev) / time_delta)  # Prevent negative rates


def fetch_latest_monitoring_data(get_db_func, release_db_func, interval=5):
    """Fetch and process monitoring data with rate calculations."""
    current_time = time.time()

    # Check if it's time for a new sample
    if monitoring_data['last_sample_time'] and \
       (current_time - monitoring_data['last_sample_time']) < interval:
        return

    conn = get_db_func()
    try:
        # Fetch current data
        top_queries = fetch_top_queries(conn)
        system_metrics = fetch_system_metrics(conn)

        # Calculate rates if we have previous samples
        if monitoring_data['previous_samples']:
            time_delta = current_time - monitoring_data['last_sample_time']

            # Process query rates
            prev_queries = monitoring_data['previous_samples'].get('top_queries', {})
            for query in top_queries:
                query_id = query['queryid']
                if query_id in prev_queries:
                    prev = prev_queries[query_id]
                    query['calls_per_sec'] = calculate_rate(
                        query['calls'], prev['calls'], time_delta
                    )

                    calls_delta = safe_float(query['calls']) - safe_float(prev['calls'])
                    if calls_delta > 0:
                        total_time_delta = (safe_float(query['total_exec_time']) -
                                          safe_float(prev['total_exec_time']))
                        query['avg_exec_time_delta'] = total_time_delta / calls_delta

                        # Calculate block rates
                        for metric in ['shared_blks_hit', 'shared_blks_read',
                                     'temp_blks_read', 'temp_blks_written']:
                            query[f'{metric}_per_sec'] = calculate_rate(
                                query.get(metric, 0), prev.get(metric, 0), time_delta
                            )
                    else:
                        query['avg_exec_time_delta'] = 0
                        for metric in ['shared_blks_hit', 'shared_blks_read',
                                     'temp_blks_read', 'temp_blks_written']:
                            query[f'{metric}_per_sec'] = 0

            # Process system metric rates
            prev_metrics = monitoring_data['previous_samples'].get('system_metrics', {})
            for category, metrics in system_metrics.items():
                if isinstance(metrics, dict) and category in prev_metrics:
                    prev = prev_metrics[category]
                    if isinstance(prev, dict):
                        # Create a list of items to avoid dict modification during iteration
                        metric_items = list(metrics.items())
                        for key, value in metric_items:
                            if isinstance(value, (int, float)) and key in prev:
                                rate_key = f"{key}_per_sec"
                                metrics[rate_key] = calculate_rate(
                                    value, prev[key], time_delta
                                )

        # Store current samples
        monitoring_data['previous_samples'] = {
            'top_queries': {q['queryid']: q for q in top_queries},
            'system_metrics': system_metrics
        }
        monitoring_data['last_sample_time'] = current_time

        # Add to history
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Store top queries
        monitoring_data['history']['top_queries'].append({
            'timestamp': timestamp,
            'queries': sorted(
                [q for q in top_queries if 'calls_per_sec' in q],
                key=lambda x: x.get('total_exec_time_delta', 0),
                reverse=True
            )[:10]
        })

        # Store system metrics with proper structure
        for category, metrics in system_metrics.items():
            if isinstance(metrics, dict):
                data_point = {'timestamp': timestamp}
                # For database metrics, we want the _per_sec values
                if category == 'database':
                    # Buffer activity uses buffer_hits and disk_reads
                    data_point['buffer_hits_per_sec'] = metrics.get('buffer_hits_per_sec', 0)
                    data_point['disk_reads_per_sec'] = metrics.get('disk_reads_per_sec', 0)
                elif category == 'io':
                    # I/O activity uses heap and index reads
                    data_point['heap_read_per_sec'] = metrics.get('heap_read_per_sec', 0)
                    data_point['idx_read_per_sec'] = metrics.get('idx_read_per_sec', 0)
                    data_point['heap_hit_per_sec'] = metrics.get('heap_hit_per_sec', 0)
                    data_point['idx_hit_per_sec'] = metrics.get('idx_hit_per_sec', 0)
                else:
                    # For other metrics, include all _per_sec values
                    per_sec_metrics = {k: v for k, v in metrics.items()
                                     if k.endswith('_per_sec') and isinstance(v, (int, float))}
                    data_point.update(per_sec_metrics)
                monitoring_data['history'][category].append(data_point)
            elif isinstance(metrics, (int, float)):
                monitoring_data['history'][category].append({
                    'timestamp': timestamp,
                    'value': metrics
                })
    finally:
        release_db_func(conn)


def fetch_top_queries(conn):
    """Fetch top queries from pg_stat_statements."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    queryid, query, calls, total_exec_time,
                    min_exec_time, max_exec_time, mean_exec_time, stddev_exec_time,
                    CASE WHEN calls > 0 THEN rows / calls ELSE 0 END AS rows,
                    shared_blks_hit, shared_blks_read, shared_blks_dirtied, shared_blks_written,
                    local_blks_hit, local_blks_read, local_blks_dirtied, local_blks_written,
                    temp_blks_read, temp_blks_written, blk_read_time, blk_write_time
                FROM pg_stat_statements
                WHERE queryid IS NOT NULL
                AND query NOT LIKE 'DO%' -- Tanel: currently ignoring the long running top level procedures
                ORDER BY total_exec_time DESC
                LIMIT 20
            """)

            columns = ['queryid', 'query', 'calls', 'total_exec_time', 'min_exec_time',
                      'max_exec_time', 'mean_exec_time', 'stddev_exec_time', 'rows',
                      'shared_blks_hit', 'shared_blks_read', 'shared_blks_dirtied',
                      'shared_blks_written', 'local_blks_hit', 'local_blks_read',
                      'local_blks_dirtied', 'local_blks_written', 'temp_blks_read',
                      'temp_blks_written', 'blk_read_time', 'blk_write_time']

            return [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching top queries: {e}")
        return []


def fetch_system_metrics(conn):
    """Fetch system-wide PostgreSQL metrics."""
    metrics = {}

    queries = {
        'database': """
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
        """,
        'bgwriter': """
            SELECT
                buffers_checkpoint,
                buffers_clean,
                buffers_backend,
                buffers_backend_fsync,
                buffers_alloc
            FROM pg_stat_bgwriter
        """,
        'io': """
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
        """
    }

    column_names = {
        'database': ['connections', 'commits', 'rollbacks', 'disk_reads', 'buffer_hits',
                    'rows_returned', 'rows_fetched', 'rows_inserted', 'rows_updated',
                    'rows_deleted'],
        'bgwriter': ['buffers_checkpoint', 'buffers_clean', 'buffers_backend',
                    'buffers_backend_fsync', 'buffers_alloc'],
        'io': ['heap_read', 'heap_hit', 'idx_read', 'idx_hit',
               'toast_read', 'toast_hit', 'tidx_read', 'tidx_hit']
    }

    with conn.cursor() as cur:
        for category, query in queries.items():
            try:
                cur.execute(query)
                row = cur.fetchone()
                if row:
                    metrics[category] = dict(
                        zip(column_names[category], [safe_float(v) for v in row])
                    )
            except Exception as e:
                print(f"Error fetching {category} metrics: {e}")
                metrics[category] = {col: 0 for col in column_names[category]}

        # Simple connection count
        try:
            cur.execute("SELECT count(*) FROM pg_stat_activity")
            metrics['connections'] = safe_float(cur.fetchone()[0])
        except Exception as e:
            print(f"Error fetching connections: {e}")
            metrics['connections'] = 0

    return metrics


def get_monitoring_data(time_range, max_samples):
    """Get monitoring data for frontend."""
    history = monitoring_data['history']

    # Get limited history
    result = {
        'top_queries': list(history['top_queries'])[-max_samples:],
        'system_metrics': {
            metric: list(data)[-max_samples:]
            for metric, data in history.items()
            if metric != 'top_queries'
        }
    }

    # Build query metrics for top 5 queries
    if result['top_queries']:
        # Find top 5 queries by total execution time
        query_totals = defaultdict(lambda: {'total_time': 0, 'query_text': ''})

        for sample in result['top_queries']:
            for query in sample.get('queries', []):
                query_id = query.get('queryid')
                if query_id:
                    query_totals[query_id]['total_time'] += query.get('total_exec_time_delta', 0)
                    query_totals[query_id]['query_text'] = query.get('query', 'Unknown Query')

        # Get top 5 query IDs
        top_5 = sorted(query_totals.items(),
                      key=lambda x: x[1]['total_time'],
                      reverse=True)[:5]
        top_5_ids = [q[0] for q in top_5]

        # Build time series for each metric
        result['query_metrics'] = {
            'execution_rates': {},
            'avg_times': {}
        }

        for query_id in top_5_ids:
            query_text = query_totals[query_id]['query_text']
            result['query_metrics']['execution_rates'][query_id] = {
                'query_text': query_text,
                'data': []
            }
            result['query_metrics']['avg_times'][query_id] = {
                'query_text': query_text,
                'data': []
            }

            # Fill time series
            for sample in result['top_queries']:
                timestamp = sample.get('timestamp', '')
                query_data = next((q for q in sample.get('queries', [])
                                 if q.get('queryid') == query_id), None)

                result['query_metrics']['execution_rates'][query_id]['data'].append({
                    'timestamp': timestamp,
                    'value': query_data.get('calls_per_sec') if query_data else None
                })
                result['query_metrics']['avg_times'][query_id]['data'].append({
                    'timestamp': timestamp,
                    'value': query_data.get('avg_exec_time_delta') if query_data else None
                })

    return result
