use super::*;
use axum::extract::Path;

#[derive(Deserialize)]
pub(super) struct Setup {
    minutes: Option<i32>,
    version: Option<i32>,
    spectate: Option<bool>,
}
#[derive(Deserialize)]
pub(super) struct Join {
    side: Option<i32>,
    invite: Option<String>,
}
#[derive(Deserialize)]
pub(super) struct Target {
    district: i32,
}
#[derive(Deserialize)]
pub(super) struct Draft {
    board: String,
    revision: i32,
}
#[derive(Deserialize)]
pub(super) struct Answer {
    board: String,
}

async fn sit(tx: &mut Tx<'_>, room: i32, team: i32, actor: i32) -> Result<(), Error> {
    let seat: Option<i32> = sqlx::query_scalar(
        "SELECT min(n)
            FROM generate_series(1, 2) n
            WHERE NOT EXISTS
                (SELECT
                 FROM match_players
                 WHERE match_id=$1
                   AND team_id=$2
                   AND seat=n)",
    )
    .bind(room)
    .bind(team)
    .fetch_one(&mut **tx)
    .await?;
    let seat = seat.ok_or_else(|| conflict("В этой команде уже два игрока"))?;
    sqlx::query("INSERT INTO team_members VALUES($1,$2) ON CONFLICT DO NOTHING")
        .bind(team)
        .bind(actor)
        .execute(&mut **tx)
        .await?;
    sqlx::query("INSERT INTO match_players VALUES($1,$2,$3,$4)")
        .bind(room)
        .bind(team)
        .bind(actor)
        .bind(seat)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
pub(super) async fn create(State(db): State<PgPool>, h: HeaderMap, Json(s): Json<Setup>) -> Api {
    let actor = user(&db, &h).await?;
    let minutes = s.minutes.unwrap_or(25);
    let spectate = s.spectate.unwrap_or(false);
    if minutes != 25 || s.version.is_some() || spectate {
        manager(&db, &h).await?;
    }
    if !(5..=60).contains(&minutes) {
        return Err(bad("Матч длится от 5 до 60 минут"));
    }
    let mut tx = db.begin().await?;
    sqlx::query("SELECT id FROM users WHERE id=$1 FOR UPDATE")
        .bind(actor)
        .fetch_one(&mut *tx)
        .await?;
    let version: i32 = sqlx::query_scalar(
        "SELECT id FROM map_versions WHERE ($1::int IS NULL OR id=$1) ORDER BY id DESC LIMIT 1",
    )
    .bind(s.version)
    .fetch_one(&mut *tx)
    .await?;
    if let Some(code) = sqlx::query_scalar::<_, String>(
        "SELECT code
            FROM matches
            WHERE host_id=$1
              AND started_at IS NULL
              AND version_id=$2
              AND minutes=$3
              AND EXISTS
                (SELECT
                 FROM match_players
                 WHERE match_id=matches.id
                   AND user_id=$1) != $4
            ORDER BY id DESC
            LIMIT 1",
    )
    .bind(actor)
    .bind(version)
    .bind(minutes)
    .bind(spectate)
    .fetch_optional(&mut *tx)
    .await?
    {
        return Ok(Json(json!({"code":code})));
    }
    let code = secret(5);
    let room: i32 = sqlx::query_scalar(
        "INSERT INTO matches(code,host_id,version_id,minutes) VALUES($1,$2,$3,$4) RETURNING id",
    )
    .bind(&code)
    .bind(actor)
    .bind(version)
    .bind(minutes)
    .fetch_one(&mut *tx)
    .await?;
    match_layout::save(&mut tx, room, version).await?;
    for (side, name) in [(0, "Синие"), (1, "Янтарные")] {
        let team: i32 = sqlx::query_scalar("INSERT INTO teams(name) VALUES($1) RETURNING id")
            .bind(name)
            .fetch_one(&mut *tx)
            .await?;
        sqlx::query("INSERT INTO match_teams VALUES($1,$2,$3)")
            .bind(room)
            .bind(team)
            .bind(side)
            .execute(&mut *tx)
            .await?;
        sqlx::query("INSERT INTO team_invites VALUES($1,$2,$3)")
            .bind(secret(6))
            .bind(room)
            .bind(team)
            .execute(&mut *tx)
            .await?;
        if side == 0 && !spectate {
            sit(&mut tx, room, team, actor).await?;
        }
    }
    sqlx::query(
        "INSERT INTO match_puzzle_pool(match_id, district_id, puzzle_id)
            SELECT $1,d.id,p.id
            FROM districts d
            JOIN campaign_stages s ON s.district_id=d.id
            JOIN puzzles example ON example.id=s.puzzle_id
            JOIN puzzles p ON p.difficulty=example.difficulty
            WHERE d.version_id=$2
              AND NOT EXISTS
                (SELECT
                 FROM campaign_stages chosen
                 WHERE chosen.puzzle_id=p.id
                   AND chosen.campaign_id=s.campaign_id)",
    )
    .bind(room)
    .bind(version)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(Json(json!({"code":code})))
}
pub(super) async fn mine(State(db): State<PgPool>, h: HeaderMap) -> Api {
    let actor = user(&db, &h).await?;
    Ok(Json(sqlx::query_scalar("SELECT COALESCE(jsonb_agg(to_jsonb(r)
                                      ORDER BY r.id DESC), '[]')
            FROM
              (SELECT m.id,m.code,v.title AS name,
                      CASE WHEN started_at IS NULL THEN 'lobby' WHEN ends_at>now() THEN 'active' ELSE 'finished'
                      END AS status
               FROM matches m
               JOIN match_players p ON p.match_id=m.id
               JOIN map_versions v ON v.id=m.version_id
               WHERE p.user_id=$1
               ORDER BY m.id DESC
               LIMIT 15) r")
        .bind(actor).fetch_one(&db).await?))
}
pub(super) async fn view(State(db): State<PgPool>, h: HeaderMap, Path(code): Path<String>) -> Api {
    let actor = user(&db, &h).await?;
    Ok(Json(
        sqlx::query_scalar(include_str!("match.sql"))
            .bind(code)
            .bind(actor)
            .fetch_one(&db)
            .await?,
    ))
}
pub(super) async fn join(
    State(db): State<PgPool>,
    h: HeaderMap,
    Path(code): Path<String>,
    Json(j): Json<Join>,
) -> Api {
    let actor = user(&db, &h).await?;
    let mut tx = db.begin().await?;
    let r =
        sqlx::query("SELECT id,started_at IS NULL AS lobby FROM matches WHERE code=$1 FOR UPDATE")
            .bind(code)
            .fetch_one(&mut *tx)
            .await?;
    let room: i32 = r.get("id");
    if sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT FROM match_players WHERE match_id=$1 AND user_id=$2)",
    )
    .bind(room)
    .bind(actor)
    .fetch_one(&mut *tx)
    .await?
    {
        return Ok(Json(json!({"joined":true})));
    }
    if !r.get::<bool, _>("lobby") {
        return Err(conflict("Матч уже начался"));
    }
    let team: i32 = if let Some(invite) = j.invite {
        sqlx::query_scalar("SELECT team_id FROM team_invites WHERE match_id=$1 AND code=$2")
            .bind(room)
            .bind(invite)
            .fetch_one(&mut *tx)
            .await?
    } else {
        sqlx::query_scalar("SELECT team_id FROM match_teams WHERE match_id=$1 AND side=$2")
            .bind(room)
            .bind(j.side.ok_or_else(|| bad("Выберите команду"))?)
            .fetch_one(&mut *tx)
            .await?
    };
    sit(&mut tx, room, team, actor).await?;
    sqlx::query(
        "UPDATE matches
            SET started_at=now(),ends_at=now()+make_interval(mins=>minutes)
            WHERE id=$1
              AND
                (SELECT count(*)
                 FROM match_players
                 WHERE match_id=$1)=4",
    )
    .bind(room)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(Json(json!({"joined":true})))
}
async fn active(tx: &mut Tx<'_>, room: i32) -> Result<(), Error> {
    if !sqlx::query_scalar::<_, bool>(
        "SELECT COALESCE(ends_at>clock_timestamp(),false) FROM matches WHERE id=$1",
    )
    .bind(room)
    .fetch_one(&mut **tx)
    .await?
    {
        return Err(conflict("Матч ещё не начался или уже завершён"));
    }
    Ok(())
}
pub(super) async fn play(
    State(db): State<PgPool>,
    h: HeaderMap,
    Path(code): Path<String>,
    Json(t): Json<Target>,
) -> Api {
    let actor = user(&db, &h).await?;
    let mut tx = db.begin().await?;
    let room: i32 = sqlx::query_scalar(
        "SELECT m.id
            FROM matches m
            JOIN match_players p ON p.match_id=m.id
            WHERE m.code=$1
              AND p.user_id=$2
              FOR
              UPDATE OF m",
    )
    .bind(code)
    .bind(actor)
    .fetch_one(&mut *tx)
    .await?;
    active(&mut tx, room).await?;
    let existing: Option<i32> = sqlx::query_scalar(
        "SELECT g.id
            FROM games g
            JOIN match_puzzle_pool p USING(match_id,district_id,puzzle_id)
            WHERE match_id=$1
              AND user_id=$2
              AND district_id=$3
              AND finished_at IS NULL
              AND p.enabled
            LIMIT 1",
    )
    .bind(room)
    .bind(actor)
    .bind(t.district)
    .fetch_optional(&mut *tx)
    .await?;
    let game = match existing {
        Some(id) => id,
        None => {
            let puzzle: i32 = sqlx::query_scalar(
                "SELECT p.puzzle_id
            FROM match_puzzle_pool p
            WHERE match_id=$1
              AND district_id=$3
              AND enabled
              AND NOT EXISTS
                (SELECT
                 FROM games g
                 WHERE g.match_id=$1
                   AND g.user_id=$2
                   AND g.puzzle_id=p.puzzle_id)
            ORDER BY random()
            LIMIT 1",
            )
            .bind(room)
            .bind(actor)
            .bind(t.district)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| conflict("В этом районе закончились задачи"))?;
            sqlx::query_scalar("INSERT INTO games(match_id,user_id,district_id,puzzle_id) VALUES($1,$2,$3,$4) RETURNING id").bind(room).bind(actor).bind(t.district).bind(puzzle).fetch_one(&mut *tx).await?
        }
    };
    sqlx::query("UPDATE games SET last_opened=clock_timestamp() WHERE id=$1")
        .bind(game)
        .execute(&mut *tx)
        .await?;
    let data=sqlx::query_scalar("SELECT jsonb_build_object('id',
                     g.id, 'givens', p.givens, 'difficulty', p.difficulty, 'board', COALESCE(m.board,
                                                                                      p.givens),
                     'revision', COALESCE(m.revision, 0))
            FROM games g
            JOIN puzzles p ON p.id=g.puzzle_id
            LEFT JOIN LATERAL
              (SELECT board,revision
               FROM game_moves
               WHERE game_id=g.id
               ORDER BY revision DESC
               LIMIT 1) m ON true
            WHERE g.id=$1").bind(game).fetch_one(&mut *tx).await?;
    tx.commit().await?;
    Ok(Json(data))
}
async fn locked_game(tx: &mut Tx<'_>, actor: i32, id: i32) -> Result<sqlx::postgres::PgRow, Error> {
    let room: i32 = sqlx::query_scalar(
        "SELECT m.id
            FROM matches m
            JOIN games g ON g.match_id=m.id
            WHERE g.id=$1
              AND g.user_id=$2
              FOR
              UPDATE OF m",
    )
    .bind(id)
    .bind(actor)
    .fetch_one(&mut **tx)
    .await?;
    let r = sqlx::query(
        "SELECT p.givens,p.solution,pool.enabled,g.finished_at IS NOT NULL AS done
            FROM games g
            JOIN puzzles p ON p.id=g.puzzle_id
            JOIN match_puzzle_pool pool USING(match_id,district_id,puzzle_id)
            WHERE g.id=$1",
    )
    .bind(id)
    .fetch_one(&mut **tx)
    .await?;
    if !r.get::<bool, _>("enabled") {
        return Err(conflict("Задача снята организатором"));
    }
    if !r.get::<bool, _>("done") {
        active(tx, room).await?;
    }
    Ok(r)
}
pub(super) async fn save(
    State(db): State<PgPool>,
    h: HeaderMap,
    Path(id): Path<i32>,
    Json(d): Json<Draft>,
) -> Api {
    let actor = user(&db, &h).await?;
    let mut tx = db.begin().await?;
    let g = locked_game(&mut tx, actor, id).await?;
    if g.get::<bool, _>("done") {
        return Err(conflict("Задача уже решена"));
    }
    if d.board.len() != 81
        || !d
            .board
            .bytes()
            .zip(g.get::<String, _>("givens").bytes())
            .all(|(v, f)| v.is_ascii_digit() && (f == b'0' || f == v))
    {
        return Err(bad("Некорректное поле"));
    }
    let rev: i32 =
        sqlx::query_scalar("SELECT COALESCE(max(revision),0) FROM game_moves WHERE game_id=$1")
            .bind(id)
            .fetch_one(&mut *tx)
            .await?;
    if rev != d.revision {
        return Err(conflict("Поле изменилось. Откройте район заново"));
    }
    sqlx::query("INSERT INTO game_moves VALUES($1,$2,$3)")
        .bind(id)
        .bind(rev + 1)
        .bind(d.board)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(Json(json!({"revision":rev+1})))
}
pub(super) async fn finish(
    State(db): State<PgPool>,
    h: HeaderMap,
    Path(id): Path<i32>,
    Json(d): Json<Answer>,
) -> Api {
    let actor = user(&db, &h).await?;
    let mut tx = db.begin().await?;
    let g = locked_game(&mut tx, actor, id).await?;
    if d.board != g.get::<String, _>("solution") {
        return Err(bad("Проверьте решение"));
    }
    if !g.get::<bool, _>("done") {
        sqlx::query("UPDATE games SET finished_at=clock_timestamp() WHERE id=$1")
            .bind(id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("INSERT INTO influence_events(game_id) VALUES($1)")
            .bind(id)
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(Json(json!({"scored":true})))
}
