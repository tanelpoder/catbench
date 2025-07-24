DO $$
  DECLARE
    e             vector(1000); -- embedding randomly picked from the table for similarity search
    r             RECORD;
    i             INT;
    minid         INT;
    maxid         INT;
    l_id          INT;
    l_count       INT;
    start_time    TIMESTAMP;

    l_file_names  VARCHAR(100)[];
    l_distances   FLOAT[];

  BEGIN
    SELECT MIN(newdogs.id), MAX(newdogs.id) INTO minid, maxid FROM newdogs;
    RAISE NOTICE 'minid=% maxid=%', minid, maxid;
    FOR i IN 1..100000 LOOP
        l_id = FLOOR(RANDOM() * (maxid - minid + 1) + minid)::INTEGER;
        SELECT embedding INTO e FROM newdogs WHERE id = l_id;
        start_time := clock_timestamp();
        
        -- Return arrays of of matching files and distances, nearest match first
        SELECT 
            ARRAY_AGG(file_name) as file_names,
            ARRAY_AGG(distance) as distances
        INTO 
            l_file_names, 
            l_distances
        FROM (
            SELECT 
                file_name,
                embedding <-> e as distance
            FROM
                newdogs
            WHERE
                id != l_id -- filter out self-match
            ORDER BY
                embedding <-> e
            LIMIT 20
        ) subq;
        
        IF i % 1000 = 0 THEN
            RAISE NOTICE 'iteration %: id=% best_distance=% time_ms=%', 
                TO_CHAR(i, '999999'),
                TO_CHAR(l_id, '9999999'),
                TO_CHAR(l_distances[1], '99.999'),
                TO_CHAR(EXTRACT(epoch FROM clock_timestamp() - start_time) * 1000, '999.99');
        END IF;
            
    END LOOP;
END $$;
