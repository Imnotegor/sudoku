use super::*;
use axum::extract::Path;
#[derive(Deserialize)]
pub(super) struct Credential {
    key: String,
}
#[derive(Deserialize)]
pub(super) struct Report {
    reason: String,
}
#[derive(Deserialize)]
pub(super) struct Decision {
    withdraw: bool,
    reason: String,
}
pub(super) async fn audit(
    tx: &mut Tx<'_>,
    actor: i32,
    action: &str,
    detail: &str,
    room: Option<i32>,
) -> Result<(), sqlx::Error> {
    sqlx::query("INSERT INTO admin_actions(actor,action,detail,match_id) VALUES($1,$2,$3,$4)")
        .bind(actor)
        .bind(action)
        .bind(detail)
        .bind(room)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
pub(super) async fn login(
    State(db): State<PgPool>,
    h: HeaderMap,
    Json(c): Json<Credential>,
) -> Api {
    let actor = user(&db, &h).await?;
    let key = std::env::var("CITY_ADMIN_KEY")
        .ok()
        .filter(|s| s.len() >= 32)
        .ok_or_else(forbidden)?;
    if hash(&c.key) != hash(&key) {
        return Err(forbidden());
    }
    let mut tx = db.begin().await?;
    sqlx::query("UPDATE users SET is_admin=true WHERE id=$1")
        .bind(actor)
        .execute(&mut *tx)
        .await?;
    audit(&mut tx, actor, "Вход администратора", "", None).await?;
    tx.commit().await?;
    profile(&db, actor).await
}
pub(super) async fn dashboard(State(db): State<PgPool>, h: HeaderMap) -> Api {
    manager(&db, &h).await?;
    Ok(Json(
        sqlx::query_scalar(include_str!("dashboard.sql"))
            .fetch_one(&db)
            .await?,
    ))
}
fn reason(s: &str) -> Result<&str, Error> {
    let v = s.trim();
    if v.is_empty() || v.chars().count() > 500 {
        Err(bad("Укажите причину длиной до 500 символов"))
    } else {
        Ok(v)
    }
}
pub(super) async fn report(
    State(db): State<PgPool>,
    h: HeaderMap,
    Path(id): Path<i32>,
    Json(r): Json<Report>,
) -> Api {
    let actor = user(&db, &h).await?;
    let can: bool = sqlx::query_scalar(
        "SELECT EXISTS
              (SELECT
               FROM games g
               JOIN match_players p ON p.match_id=g.match_id
               WHERE g.id=$1
                 AND p.user_id=$2)
            OR
              (SELECT is_admin
               FROM users
               WHERE id=$2)",
    )
    .bind(id)
    .bind(actor)
    .fetch_one(&db)
    .await?;
    if !can {
        return Err(Error(
            StatusCode::FORBIDDEN,
            "Жалобы доступны участникам матча",
        ));
    }
    sqlx::query("INSERT INTO match_reports(game_id,reporter,reason) VALUES($1,$2,$3) ON CONFLICT DO NOTHING").bind(id).bind(actor).bind(reason(&r.reason)?).execute(&db).await?;
    Ok(Json(json!({"reported":true})))
}
pub(super) async fn resolve(
    State(db): State<PgPool>,
    h: HeaderMap,
    Path(id): Path<i32>,
    Json(d): Json<Decision>,
) -> Api {
    let actor = manager(&db, &h).await?;
    let detail = reason(&d.reason)?;
    let mut tx = db.begin().await?;
    let r = sqlx::query(
        "SELECT g.match_id,g.puzzle_id,r.resolved
            FROM match_reports r
            JOIN games g ON g.id=r.game_id
            JOIN matches m ON m.id=g.match_id
            WHERE r.id=$1
              FOR
              UPDATE OF m,r",
    )
    .bind(id)
    .fetch_one(&mut *tx)
    .await?;
    if r.get::<bool, _>("resolved") {
        return Err(conflict("Решение уже принято"));
    }
    let room: i32 = r.get("match_id");
    let puzzle: i32 = r.get("puzzle_id");
    if d.withdraw {
        sqlx::query(
            "UPDATE match_puzzle_pool SET enabled=false WHERE match_id=$1 AND puzzle_id=$2",
        )
        .bind(room)
        .bind(puzzle)
        .execute(&mut *tx)
        .await?;
    }
    sqlx::query("UPDATE match_reports SET resolved=true WHERE id=$1")
        .bind(id)
        .execute(&mut *tx)
        .await?;
    audit(
        &mut tx,
        actor,
        if d.withdraw {
            "Задача снята"
        } else {
            "Жалоба закрыта"
        },
        &format!("№{puzzle} · {detail}"),
        Some(room),
    )
    .await?;
    tx.commit().await?;
    Ok(Json(json!({"resolved":true})))
}
