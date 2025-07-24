* **catbench** is an app that uses Postgres+pgvector as a backend and you need to use my setup scripts to load data into it first

# PostgreSQL Performance Monitoring for CatBench

The CatBench application includes a real-time monitoring dashboard for PostgreSQL performance. This feature allows you to visualize:

1. Query performance metrics from `pg_stat_statements`
2. System-wide database metrics
3. I/O and buffer activity
4. Connection counts

## How to Use the Monitoring Dashboard

1. Navigate to the "Database Monitoring" section on the CatBench homepage
2. Click on "Go to Monitoring" to access the dashboard
3. Use the refresh interval dropdown to control how often data is updated
4. Click "Refresh Now" to manually update the metrics

## Running a Workload with cat_loop.sql

To see the monitoring in action, you can run one of the included workload scripts in a separate terminal:

```bash
# Connect to your PostgreSQL database
psql -d your_database_name

# Run the cat_loop.sql workload
\i path/to/cat_loop.sql
```

## Requirements

The monitoring dashboard requires:

1. The `pg_stat_statements` extension must be enabled in your PostgreSQL instance
2. The user connecting to the database must have appropriate permissions to query system catalogs

To enable `pg_stat_statements` (if not already enabled):

```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

You may need to add the following to your `postgresql.conf` file and restart PostgreSQL:

```
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
pg_stat_statements.max = 10000
```

## Implementation Details

The monitoring system:

1. Takes periodic snapshots of PostgreSQL performance metrics
2. Calculates deltas between samples to show rates rather than cumulative values
3. Stores a history of recent samples for timeline visualization
4. Automatically updates charts with fresh data at configurable intervals

All monitoring data is stored in memory only, and is reset when the application restarts.

