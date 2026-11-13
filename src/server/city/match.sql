SELECT jsonb_build_object(
 'code',m.code,'minutes',m.minutes,'host',m.host_id,'map',v.title,'me',$2::int,
 'status',CASE WHEN m.started_at IS NULL THEN 'lobby' WHEN m.ends_at<=now() THEN 'finished' ELSE 'active' END,
 'ends',EXTRACT(epoch FROM m.ends_at)::bigint,'now',EXTRACT(epoch FROM now())::bigint,
 'teams',(SELECT jsonb_agg(jsonb_build_object('side',t.side,'name',teams.name,'invite',i.code,
   'players',COALESCE((SELECT jsonb_agg(jsonb_build_object('id',u.id,'name',u.name,'district',(SELECT CASE WHEN g.finished_at IS NULL AND m.ends_at>now() THEN d.name END FROM games g JOIN districts d ON d.id=g.district_id WHERE g.user_id=u.id AND g.match_id=m.id ORDER BY g.last_opened DESC LIMIT 1)) ORDER BY mp.seat)
      FROM match_players mp JOIN users u ON u.id=mp.user_id WHERE mp.match_id=m.id AND mp.team_id=t.team_id),'[]')) ORDER BY t.side)
   FROM match_teams t JOIN teams ON teams.id=t.team_id JOIN team_invites i USING(match_id,team_id) WHERE t.match_id=m.id),
 'districts',(SELECT jsonb_agg(jsonb_build_object('id',d.id,'slot',d.slot,'name',d.name,
   'x',COALESCE((m.layout->'positions'->(d.id::text)->>0)::int,d.x),
   'y',COALESCE((m.layout->'positions'->(d.id::text)->>1)::int,d.y),
   'scores',jsonb_build_array(COALESCE((SELECT points FROM district_scores WHERE match_id=m.id AND district_id=d.id AND side=0),0),
                              COALESCE((SELECT points FROM district_scores WHERE match_id=m.id AND district_id=d.id AND side=1),0)))
   ORDER BY d.slot) FROM districts d WHERE d.version_id=m.version_id),
 'links',COALESCE(m.layout->'links',(SELECT jsonb_agg(jsonb_build_array(a,b) ORDER BY a,b) FROM district_links WHERE version_id=m.version_id)),
 'events',COALESCE((SELECT jsonb_agg(to_jsonb(k) ORDER BY k.scored_at DESC) FROM
   (SELECT g.id,e.scored_at,u.name AS player,d.name AS district,mp.district_id,mt.side,mp.enabled,g.puzzle_id
    FROM influence_events e JOIN games g ON g.id=e.game_id JOIN users u ON u.id=g.user_id
    JOIN districts d ON d.id=g.district_id JOIN match_puzzle_pool mp USING(match_id,district_id,puzzle_id)
    JOIN match_players pl ON pl.user_id=g.user_id AND pl.match_id=g.match_id
    JOIN match_teams mt ON mt.team_id=pl.team_id AND mt.match_id=g.match_id
    WHERE g.match_id=m.id ORDER BY e.scored_at DESC) k),'[]'))
FROM matches m JOIN map_versions v ON v.id=m.version_id WHERE m.code=$1
