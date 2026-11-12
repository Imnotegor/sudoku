mod admin;
mod battle;
mod map;

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use rand::{rngs::OsRng, RngCore};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{postgres::PgPoolOptions, PgPool, Postgres, Row, Transaction};

type Tx<'a> = Transaction<'a, Postgres>;
type Api = Result<Json<Value>, Error>;
struct Error(StatusCode, &'static str);
impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":self.1}))).into_response()
    }
}
impl From<sqlx::Error> for Error {
    fn from(e: sqlx::Error) -> Self {
        if matches!(e, sqlx::Error::RowNotFound) {
            return Self(StatusCode::NOT_FOUND, "Запись не найдена");
        }
        tracing::error!(error=%e,"city database");
        Self(StatusCode::INTERNAL_SERVER_ERROR, "Ошибка базы данных")
    }
}
fn bad(msg: &'static str) -> Error {
    Error(StatusCode::BAD_REQUEST, msg)
}
fn conflict(msg: &'static str) -> Error {
    Error(StatusCode::CONFLICT, msg)
}
fn forbidden() -> Error {
    Error(StatusCode::FORBIDDEN, "Требуются права администратора")
}
fn secret(bytes: usize) -> String {
    let mut buf = vec![0; bytes];
    OsRng.fill_bytes(&mut buf);
    buf.iter().map(|b| format!("{b:02x}")).collect()
}
fn hash(s: &str) -> String {
    format!("{:x}", Sha256::digest(s.as_bytes()))
}
fn token(h: &HeaderMap) -> Option<&str> {
    h.get(header::COOKIE)?
        .to_str()
        .ok()?
        .split(';')
        .find_map(|c| c.trim().strip_prefix("sudoku_session="))
        .filter(|s| s.len() == 64)
}
async fn user(db: &PgPool, h: &HeaderMap) -> Result<i32, Error> {
    let unauth = || Error(StatusCode::UNAUTHORIZED, "Откройте профиль игрока");
    sqlx::query_scalar("SELECT user_id FROM auth_sessions WHERE token_hash=$1 AND expires_at>now()")
        .bind(hash(token(h).ok_or_else(unauth)?))
        .fetch_optional(db)
        .await?
        .ok_or_else(unauth)
}
async fn manager(db: &PgPool, h: &HeaderMap) -> Result<i32, Error> {
    let id = user(db, h).await?;
    if sqlx::query_scalar::<_, bool>("SELECT is_admin FROM users WHERE id=$1")
        .bind(id)
        .fetch_one(db)
        .await?
    {
        Ok(id)
    } else {
        Err(forbidden())
    }
}
async fn profile(db: &PgPool, id: i32) -> Api {
    Ok(Json(sqlx::query_scalar("SELECT jsonb_build_object('id',id,'name',name,'admin',is_admin) FROM users WHERE id=$1")
        .bind(id).fetch_one(db).await?))
}
async fn session(State(db): State<PgPool>, h: HeaderMap) -> Result<Response, Error> {
    if token(&h).is_some() {
        match user(&db, &h).await {
            Ok(id) => return Ok(profile(&db, id).await?.into_response()),
            Err(Error(StatusCode::UNAUTHORIZED, _)) => (),
            Err(e) => return Err(e),
        }
    }
    let mut tx = db.begin().await?;
    let id: i32 = sqlx::query_scalar("INSERT INTO users(name) VALUES($1) RETURNING id")
        .bind(format!("Игрок {}", secret(2)))
        .fetch_one(&mut *tx)
        .await?;
    let token = secret(32);
    sqlx::query("INSERT INTO auth_sessions VALUES($1,$2,now()+interval '180 days')")
        .bind(hash(&token))
        .bind(id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    let secure = if std::env::var("CITY_SECURE_COOKIE").as_deref() == Ok("1") {
        "; Secure"
    } else {
        ""
    };
    let cookie =
        format!("sudoku_session={token}; Path=/; Max-Age=15552000; HttpOnly; SameSite=Lax{secure}");
    Ok(([(header::SET_COOKIE, cookie)], profile(&db, id).await?).into_response())
}
#[derive(Deserialize)]
struct Name {
    name: String,
}
async fn rename(State(db): State<PgPool>, h: HeaderMap, Json(n): Json<Name>) -> Api {
    let id = user(&db, &h).await?;
    let name = n.name.trim();
    if name.is_empty() || name.chars().count() > 32 {
        return Err(bad("Введите имя от 1 до 32 символов"));
    }
    sqlx::query("UPDATE users SET name=$1 WHERE id=$2")
        .bind(name)
        .bind(id)
        .execute(&db)
        .await?;
    profile(&db, id).await
}
async fn private_api(req: Request, next: Next) -> Response {
    if req.method() != axum::http::Method::GET
        && (req
            .headers()
            .get("x-city-request")
            .and_then(|v| v.to_str().ok())
            != Some("1")
            || req
                .headers()
                .get("sec-fetch-site")
                .and_then(|v| v.to_str().ok())
                == Some("cross-site"))
    {
        return Error(StatusCode::FORBIDDEN, "Недопустимый запрос").into_response();
    }
    let mut res = next.run(req).await;
    res.headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    res
}
async fn health(State(db): State<PgPool>) -> Api {
    let ready: bool = sqlx::query_scalar("SELECT EXISTS (SELECT FROM campaigns)")
        .fetch_one(&db)
        .await?;
    Ok(Json(json!({"ready": ready})))
}

pub async fn router(url: &str) -> Result<Router, sqlx::Error> {
    let db = PgPoolOptions::new()
        .max_connections(6)
        .after_connect(|c, _| {
            Box::pin(async move {
                sqlx::query("SET search_path TO city, pg_catalog")
                    .execute(c)
                    .await?;
                Ok(())
            })
        })
        .connect(url)
        .await?;
    let mut tx = db.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock(229046501)")
        .execute(&mut *tx)
        .await?;
    if !sqlx::query_scalar::<_, bool>("SELECT to_regclass('city.users') IS NOT NULL")
        .fetch_one(&mut *tx)
        .await?
    {
        sqlx::query("CREATE SCHEMA IF NOT EXISTS city")
            .execute(&mut *tx)
            .await?;
        sqlx::raw_sql(include_str!("schema.sql"))
            .execute(&mut *tx)
            .await?;
        sqlx::raw_sql(include_str!("puzzles.sql"))
            .execute(&mut *tx)
            .await?;
        map::seed(&mut tx).await?;
    }
    tx.commit().await?;
    Ok(Router::new()
        .route("/health", get(health))
        .route("/session", post(session))
        .route("/profile", post(rename))
        .route("/map", get(map::current))
        .route("/campaign/:id/finish", post(map::finish))
        .route("/matches", post(battle::create).get(battle::mine))
        .route("/matches/:code", get(battle::view))
        .route("/matches/:code/join", post(battle::join))
        .route("/matches/:code/play", post(battle::play))
        .route("/games/:id/move", post(battle::save))
        .route("/games/:id/finish", post(battle::finish))
        .route("/games/:id/report", post(admin::report))
        .route("/admin/login", post(admin::login))
        .route("/admin", get(admin::dashboard))
        .route("/admin/maps", post(map::publish))
        .route("/admin/reports/:id", post(admin::resolve))
        .layer(middleware::from_fn(private_api))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .with_state(db))
}
