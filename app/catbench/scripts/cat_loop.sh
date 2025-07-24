psql tpcc -f ./cat_loop.sql &
psql tpcc -f ./cat_loop.sql &
psql tpcc -f ./cat_loop.sql &
psql tpcc -f ./cat_loop.sql &

# If using a remote postgres instance over network, edit and uncomment the lines below:

# export PGPASSWORD=tpcc
# 
# psql -h localhost -U tpcc tpcc -f ./cat_loop.sql &
# psql -h localhost -U tpcc tpcc -f ./cat_loop.sql &
# psql -h localhost -U tpcc tpcc -f ./cat_loop.sql &
# psql -h localhost -U tpcc tpcc -f ./cat_loop.sql &

# uncomment more lines below if you want more concurrent workload

# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &
# psql tpcc -f ./cat_loop.sql &

# kill all child processes on CTRL+C
wait
