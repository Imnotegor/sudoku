SELECT jsonb_build_object('map',
                            (SELECT to_jsonb(m)
                             FROM city_maps m
                             ORDER BY version_id DESC
                             LIMIT 1), 'versions',
                            (SELECT jsonb_agg(to_jsonb(v)
                                              ORDER BY version_id DESC)
                             FROM
                               (SELECT version_id, name
                                FROM city_maps) v), 'puzzles',
                            (SELECT jsonb_agg(jsonb_build_object('id',
                                                id, 'difficulty', difficulty)
                                              ORDER BY id)
                             FROM puzzles), 'matches',

         COALESCE(
                    (SELECT jsonb_agg(to_jsonb(x))
                     FROM
                       (SELECT m.code,
                 v.title AS name, m.minutes,
                 CASE WHEN m.started_at IS NULL THEN 'lobby' WHEN m.ends_at>now() THEN 'active' ELSE 'finished'
                 END AS status
                        FROM matches m
                        JOIN map_versions v ON v.id=m.version_id
                        ORDER BY m.id DESC
                        LIMIT 60)x),'[]'), 'reports',
         COALESCE(
                    (SELECT jsonb_agg(to_jsonb(x))
                     FROM
                       (SELECT r.id,
                 r.reason, r.resolved, g.puzzle_id, m.code, d.name AS district, u.name AS player
                        FROM match_reports r
                        JOIN games g ON g.id=r.game_id
                        JOIN matches m ON m.id=g.match_id
                        JOIN districts d ON d.id=g.district_id
                        JOIN users u ON u.id=r.reporter
                        ORDER BY r.resolved, r.id DESC
                        LIMIT 100)x),'[]'), 'actions',
         COALESCE(
                    (SELECT jsonb_agg(to_jsonb(x))
                     FROM
                       (SELECT a.action, a.detail, u.name AS actor, a.created_at
                        FROM admin_actions a
                        JOIN users u ON u.id=a.actor
                        ORDER BY a.id DESC
                        LIMIT 50)x),'[]'))
