CREATE TABLE users (
 id serial PRIMARY KEY, name varchar(32) NOT NULL, is_admin boolean NOT NULL DEFAULT false
);
CREATE TABLE auth_sessions (
 token_hash text PRIMARY KEY, user_id integer NOT NULL REFERENCES users, expires_at timestamptz NOT NULL
);
CREATE TABLE teams (id serial PRIMARY KEY, name text NOT NULL);
CREATE TABLE team_members (
 team_id integer REFERENCES teams, user_id integer REFERENCES users, PRIMARY KEY(team_id,user_id)
);
CREATE TABLE puzzles (
 id serial PRIMARY KEY, givens text UNIQUE NOT NULL CHECK(givens ~ '^[0-9]{81}$'),
 solution text NOT NULL CHECK(solution ~ '^[1-9]{81}$'), difficulty text NOT NULL
);
CREATE TABLE maps (id serial PRIMARY KEY, name text NOT NULL);
CREATE TABLE map_versions (id serial PRIMARY KEY, map_id integer NOT NULL REFERENCES maps, title text NOT NULL, published_at timestamptz DEFAULT now());
CREATE TABLE districts (
 id serial PRIMARY KEY, version_id integer NOT NULL REFERENCES map_versions,
 slot smallint NOT NULL CHECK(slot BETWEEN 1 AND 5), name text NOT NULL,
 x int NOT NULL CHECK(x BETWEEN 0 AND 100), y int NOT NULL CHECK(y BETWEEN 0 AND 100),
 UNIQUE(version_id,slot), UNIQUE(version_id,id)
);
CREATE TABLE district_links (
 version_id integer NOT NULL REFERENCES map_versions, a integer NOT NULL, b integer NOT NULL,
 CHECK(a < b), PRIMARY KEY(version_id,a,b),
 FOREIGN KEY(version_id,a) REFERENCES districts(version_id,id),
 FOREIGN KEY(version_id,b) REFERENCES districts(version_id,id)
);
CREATE TABLE campaigns (id serial PRIMARY KEY, version_id integer UNIQUE NOT NULL REFERENCES map_versions);
CREATE TABLE campaign_stages (
 campaign_id integer REFERENCES campaigns, district_id integer REFERENCES districts,
 puzzle_id integer NOT NULL REFERENCES puzzles, is_start boolean NOT NULL, is_finish boolean NOT NULL,
 PRIMARY KEY(campaign_id,district_id)
);
CREATE TABLE campaign_progress (
 user_id integer REFERENCES users, campaign_id integer NOT NULL REFERENCES campaigns, district_id integer NOT NULL,
 completed_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(user_id,campaign_id,district_id),
 FOREIGN KEY(campaign_id,district_id) REFERENCES campaign_stages
);
CREATE TABLE matches (
 id serial PRIMARY KEY, code text UNIQUE NOT NULL, host_id integer NOT NULL REFERENCES users,
 version_id integer NOT NULL REFERENCES map_versions, minutes integer NOT NULL CHECK(minutes BETWEEN 5 AND 60),
 created_at timestamptz NOT NULL DEFAULT now(), started_at timestamptz, ends_at timestamptz
);
CREATE TABLE match_teams (
 match_id integer REFERENCES matches, team_id integer REFERENCES teams, side int NOT NULL CHECK(side IN (0,1)),
 PRIMARY KEY(match_id,team_id), UNIQUE(match_id,side)
);
CREATE TABLE team_invites (
 code text PRIMARY KEY, match_id integer NOT NULL, team_id integer NOT NULL,
 FOREIGN KEY(match_id,team_id) REFERENCES match_teams, UNIQUE(match_id,team_id)
);
CREATE TABLE match_players (
 match_id integer NOT NULL, team_id integer NOT NULL, user_id integer NOT NULL, seat int NOT NULL CHECK(seat IN (1,2)),
 PRIMARY KEY(match_id,user_id), UNIQUE(match_id,team_id,seat),
 FOREIGN KEY(match_id,team_id) REFERENCES match_teams, FOREIGN KEY(team_id,user_id) REFERENCES team_members
);
CREATE TABLE match_puzzle_pool (
 match_id integer REFERENCES matches, district_id integer REFERENCES districts, puzzle_id integer REFERENCES puzzles,
 enabled boolean NOT NULL DEFAULT true, PRIMARY KEY(match_id,district_id,puzzle_id)
);
CREATE TABLE games (
 id serial PRIMARY KEY, match_id integer NOT NULL, user_id integer NOT NULL, district_id integer NOT NULL, puzzle_id integer NOT NULL,
 last_opened timestamptz NOT NULL DEFAULT clock_timestamp(), created_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz,
 FOREIGN KEY(match_id,user_id) REFERENCES match_players, FOREIGN KEY(match_id,district_id,puzzle_id) REFERENCES match_puzzle_pool,
 UNIQUE(match_id,user_id,puzzle_id)
);
CREATE TABLE game_moves (
 game_id integer REFERENCES games, revision int NOT NULL, board text NOT NULL CHECK(board ~ '^[0-9]{81}$'),
 PRIMARY KEY(game_id,revision)
);
CREATE TABLE influence_events (
 game_id integer PRIMARY KEY REFERENCES games, scored_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE match_reports (
 id serial PRIMARY KEY, game_id integer NOT NULL REFERENCES games, reporter integer NOT NULL REFERENCES users,
 reason text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), resolved boolean NOT NULL DEFAULT false,
 UNIQUE(game_id,reporter)
);
CREATE TABLE admin_actions (
 id serial PRIMARY KEY, actor integer NOT NULL REFERENCES users, action text NOT NULL, detail text NOT NULL,
 match_id integer REFERENCES matches, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX games_player_district ON games(match_id,user_id,district_id,id DESC);
CREATE INDEX progress_user ON campaign_progress(user_id,campaign_id);
CREATE VIEW district_scores AS
 SELECT g.match_id,g.district_id,mt.side,count(*)::int AS points
 FROM influence_events e JOIN games g ON g.id=e.game_id
 JOIN match_players mp USING(match_id,user_id) JOIN match_teams mt USING(match_id,team_id)
 JOIN match_puzzle_pool pool USING(match_id,district_id,puzzle_id)
 WHERE pool.enabled GROUP BY g.match_id,g.district_id,mt.side;

CREATE VIEW city_maps AS
 SELECT v.id AS version_id,c.id AS campaign_id,v.title AS name,
   (SELECT jsonb_agg(jsonb_build_object('id',d.id,'slot',d.slot,'name',d.name,'x',d.x,'y',d.y,
         'start',s.is_start,'finish',s.is_finish,'puzzle',s.puzzle_id,'givens',p.givens,'difficulty',p.difficulty) ORDER BY d.slot)
    FROM districts d JOIN campaign_stages s ON s.district_id=d.id JOIN puzzles p ON p.id=s.puzzle_id
    WHERE d.version_id=v.id) AS districts,
   (SELECT jsonb_agg(jsonb_build_array(a,b) ORDER BY a,b) FROM district_links l WHERE l.version_id=v.id) AS links
 FROM map_versions v JOIN maps m ON m.id=v.map_id JOIN campaigns c ON c.version_id=v.id;
