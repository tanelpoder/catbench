-- Cat loop with recall tracking
-- This script sends pg_notify messages every 1000 iterations
-- The DO block can use COMMIT since it's not wrapped in an explicit transaction

DO $$
DECLARE
    e             vector(1000);
    i             INT;
    minid         INT;
    maxid         INT;
    l_id          INT;
    start_time    TIMESTAMP;
    l_distance    FLOAT;
    
    -- Variables for fraud detection recall tracking
    total_attempts INT := 0;
    successful_recalls INT := 0;
    recall_percentage NUMERIC(5,2);
    new_file_name VARCHAR(100);
    closest_match_name VARCHAR(100);
    expected_base_name VARCHAR(100);
    is_symmetric_match BOOLEAN;
    
    -- JSON payload for pg_notify
    notify_payload JSON;
    
BEGIN
    SELECT MIN(id), MAX(id) INTO minid, maxid FROM newcats;
    RAISE NOTICE 'newcats minid=% maxid=%', minid, maxid;
    
    FOR i IN 1..100000 LOOP
        -- Select a random "new customer photo" from newcats
        l_id = FLOOR(RANDOM() * (maxid - minid + 1) + minid)::INTEGER;
        SELECT file_name, embedding INTO new_file_name, e FROM newcats WHERE id = l_id;
        
        start_time := clock_timestamp();
        
        -- Find the closest match in the original cats table
        SELECT 
            file_name,
            embedding <-> e as distance
        INTO 
            closest_match_name,
            l_distance
        FROM
            cats
        ORDER BY
            embedding <-> e
        LIMIT 1;
        
        -- Extract base filename from the rotated variant
        expected_base_name := REGEXP_REPLACE(new_file_name, '^\d+_', '');
        
        -- Check if the closest match is a symmetric match
        is_symmetric_match := (closest_match_name = expected_base_name);
        
        -- Update recall tracking
        total_attempts := total_attempts + 1;
        IF is_symmetric_match THEN
            successful_recalls := successful_recalls + 1;
        END IF;
        
        -- Log progress and send notifications every 1000 iterations
        IF i % 1000 = 0 THEN
            recall_percentage := CASE 
                WHEN total_attempts > 0 THEN (successful_recalls::NUMERIC / total_attempts * 100)::NUMERIC(5,2)
                ELSE 0.00
            END;
            
            RAISE NOTICE 'iteration %: attempts=% recalls=% percentage=% example: new=% -> found=% (match=%)', 
                TO_CHAR(i, '999999'),
                total_attempts,
                successful_recalls,
                recall_percentage || '%',
                new_file_name,
                closest_match_name,
                CASE WHEN is_symmetric_match THEN 'YES' ELSE 'NO' END;
            
            -- Create JSON payload for pg_notify
            notify_payload := json_build_object(
                'script_name', 'cat_loop',
                'timestamp', to_char(NOW(), 'HH24:MI:SS'),
                'total_attempts', total_attempts,
                'successful_recalls', successful_recalls,
                'recall_percentage', recall_percentage
            );
            
            -- Send notification
            PERFORM pg_notify('catbench_recall', notify_payload::text);
            
            -- Commit transaction to send notifications immediately
            COMMIT;
            
            -- Reset counters for next batch
            total_attempts := 0;
            successful_recalls := 0;
        END IF;
    END LOOP;
    
    -- Send final notification if there are remaining attempts
    IF total_attempts > 0 THEN
        recall_percentage := CASE 
            WHEN total_attempts > 0 THEN (successful_recalls::NUMERIC / total_attempts * 100)::NUMERIC(5,2)
            ELSE 0.00
        END;
        
        notify_payload := json_build_object(
            'script_name', 'cat_loop',
            'timestamp', to_char(NOW(), 'HH24:MI:SS'),
            'total_attempts', total_attempts,
            'successful_recalls', successful_recalls,
            'recall_percentage', recall_percentage
        );
        
        PERFORM pg_notify('catbench_recall', notify_payload::text);
        COMMIT;
    END IF;
    
    RAISE NOTICE 'Cat loop completed';
END $$;