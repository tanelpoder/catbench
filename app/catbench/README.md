## Requirements for CatBench Monitoring

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

