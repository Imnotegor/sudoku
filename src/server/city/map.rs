use super::*;
use axum::extract::{Path, Query};
use std::collections::HashSet;

#[derive(Deserialize)]
pub(super) struct District {
    name: String,
    x: i32,
    y: i32,
    puzzle: i32,
}
#[derive(Deserialize)]
pub(super) struct Layout {
    name: String,
    districts: Vec<District>,
    links: Vec<[usize; 2]>,
    start: usize,
    finish: usize,
    #[serde(default)]
    base: i32,
}
impl Layout {
    fn validate(&self) -> Result<(), Error> {
        if self.districts.len() != 5
            || self.start >= 5
            || self.finish >= 5
            || self.start == self.finish
            || self.name.trim().is_empty()
            || self.name.chars().count() > 60
            || self.districts.iter().any(|d| {
                d.name.trim().is_empty()
                    || d.name.chars().count() > 32
                    || !(8..=92).contains(&d.x)
                    || !(8..=92).contains(&d.y)
            })
        {
            return Err(bad(
                "Проверьте название, пять районов, координаты, старт и финиш",
            ));
        }
        let mut edges = HashSet::new();
        for &[a, b] in &self.links {
            if a >= 5 || b >= 5 || a == b || !edges.insert((a.min(b), a.max(b))) {
                return Err(bad("Некорректная дорога"));
            }
        }
        let mut seen = HashSet::from([self.start]);
        for _ in 0..5 {
            for &(a, b) in &edges {
                if seen.contains(&a) || seen.contains(&b) {
                    seen.insert(a);
                    seen.insert(b);
                }
            }
        }
        if seen.len() != 5 {
            return Err(bad("Каждый район должен быть достижим из стартового"));
        }
        Ok(())
    }
    async fn insert(&self, tx: &mut Tx<'_>, map: i32) -> Result<i32, sqlx::Error> {
        let version: i32 =
            sqlx::query_scalar("INSERT INTO map_versions(map_id,title) VALUES($1,$2) RETURNING id")
                .bind(map)
                .bind(self.name.trim())
                .fetch_one(&mut **tx)
                .await?;
        let campaign: i32 =
            sqlx::query_scalar("INSERT INTO campaigns(version_id) VALUES($1) RETURNING id")
                .bind(version)
                .fetch_one(&mut **tx)
                .await?;
        let mut ids = Vec::new();
        for (i, d) in self.districts.iter().enumerate() {
            let id:i32=sqlx::query_scalar("INSERT INTO districts(version_id,slot,name,x,y) VALUES($1,$2,$3,$4,$5) RETURNING id")
                .bind(version).bind((i+1) as i16).bind(d.name.trim()).bind(d.x).bind(d.y).fetch_one(&mut **tx).await?;
            sqlx::query("INSERT INTO campaign_stages VALUES($1,$2,$3,$4,$5)")
                .bind(campaign)
                .bind(id)
                .bind(d.puzzle)
                .bind(i == self.start)
                .bind(i == self.finish)
                .execute(&mut **tx)
                .await?;
            ids.push(id);
        }
        for &[a, b] in &self.links {
            sqlx::query("INSERT INTO district_links VALUES($1,$2,$3)")
                .bind(version)
                .bind(ids[a].min(ids[b]))
                .bind(ids[a].max(ids[b]))
                .execute(&mut **tx)
                .await?;
        }
        Ok(version)
    }
}
pub(super) async fn seed(tx: &mut Tx<'_>) -> Result<(), sqlx::Error> {
    let map = sqlx::query_scalar("INSERT INTO maps(name) VALUES('Речной город') RETURNING id")
        .fetch_one(&mut **tx)
        .await?;
    let districts = [
        ("Вокзал", 17, 55, 1),
        ("Парк", 40, 22, 2),
        ("Университет", 48, 76, 49),
        ("Порт", 68, 42, 25),
        ("Маяк", 86, 70, 26),
    ]
    .into_iter()
    .map(|(name, x, y, puzzle)| District {
        name: name.into(),
        x,
        y,
        puzzle,
    })
    .collect();
    Layout {
        name: "Речной город".into(),
        districts,
        links: vec![[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 4]],
        start: 0,
        finish: 4,
        base: 0,
    }
    .insert(tx, map)
    .await?;
    Ok(())
}
pub(super) async fn publish(
    State(db): State<PgPool>,
    h: HeaderMap,
    Json(layout): Json<Layout>,
) -> Api {
    let actor = manager(&db, &h).await?;
    layout.validate()?;
    let mut tx = db.begin().await?;
    sqlx::query("SELECT id FROM maps WHERE id=1 FOR UPDATE")
        .fetch_one(&mut *tx)
        .await?;
    let current: i32 = sqlx::query_scalar("SELECT max(id) FROM map_versions WHERE map_id=1")
        .fetch_one(&mut *tx)
        .await?;
    if layout.base != current {
        return Err(conflict("Карта обновилась. Загрузите последнюю версию"));
    }
    for d in &layout.districts {
        if !sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT FROM puzzles WHERE id=$1)")
            .bind(d.puzzle)
            .fetch_one(&mut *tx)
            .await?
        {
            return Err(bad("Задача района отсутствует"));
        }
    }
    let version = layout.insert(&mut tx, 1).await?;
    admin::audit(
        &mut tx,
        actor,
        "Публикация карты",
        &format!("{} · версия {version}", layout.name.trim()),
        None,
    )
    .await?;
    tx.commit().await?;
    Ok(Json(json!({"version":version})))
}
#[derive(Deserialize)]
pub(super) struct Select {
    campaign: Option<i32>,
}
pub(super) async fn current(
    State(db): State<PgPool>,
    h: HeaderMap,
    Query(select): Query<Select>,
) -> Api {
    let actor = user(&db, &h).await?;
    Ok(Json(
        sqlx::query_scalar(
            "SELECT to_jsonb(m) || jsonb_build_object('done',

                                    ARRAY
                                                       (SELECT district_id
                                                        FROM campaign_progress
                                                        WHERE user_id=$1
                                                          AND campaign_id=m.campaign_id))
            FROM city_maps m
            WHERE ($2::int IS NULL
                   OR m.campaign_id=$2)
            ORDER BY version_id DESC
            LIMIT 1",
        )
        .bind(actor)
        .bind(select.campaign)
        .fetch_one(&db)
        .await?,
    ))
}
#[derive(Deserialize)]
pub(super) struct Completion {
    district: i32,
    board: String,
}
pub(super) async fn finish(
    State(db): State<PgPool>,
    h: HeaderMap,
    Path(campaign): Path<i32>,
    Json(data): Json<Completion>,
) -> Api {
    let actor = user(&db, &h).await?;
    let mut tx = db.begin().await?;
    sqlx::query("SELECT id FROM users WHERE id=$1 FOR UPDATE")
        .bind(actor)
        .fetch_one(&mut *tx)
        .await?;
    let row = sqlx::query(
        "SELECT p.solution,s.is_start
            OR EXISTS
              (SELECT
               FROM campaign_progress cp
               JOIN district_links l ON (l.a=cp.district_id
                                         AND l.b=s.district_id)
               OR (l.b=cp.district_id
                   AND l.a=s.district_id)
               WHERE cp.user_id=$3
                 AND cp.campaign_id=$1) AS unlocked
            FROM campaign_stages s
            JOIN puzzles p ON p.id=s.puzzle_id
            WHERE s.campaign_id=$1
              AND s.district_id=$2",
    )
    .bind(campaign)
    .bind(data.district)
    .bind(actor)
    .fetch_one(&mut *tx)
    .await?;
    if !row.get::<bool, _>("unlocked") {
        return Err(bad("Район ещё закрыт"));
    }
    if data.board != row.get::<String, _>("solution") {
        return Err(bad("Проверьте решение"));
    }
    sqlx::query("INSERT INTO campaign_progress(user_id,campaign_id,district_id) VALUES($1,$2,$3) ON CONFLICT DO NOTHING")
        .bind(actor).bind(campaign).bind(data.district).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(Json(json!({"completed":true})))
}
