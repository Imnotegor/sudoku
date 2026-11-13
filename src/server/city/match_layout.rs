use super::Tx;
use rand::{seq::SliceRandom, Rng};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Serialize)]
struct MatchLayout {
    positions: BTreeMap<i32, [i32; 2]>,
    links: Vec<[i32; 2]>,
}

// Slots run clockwise. Convex outlines keep the perimeter and diagonals
// separate even after the small offsets below. All coordinates are percentages.
const RING: [[i32; 2]; 5] = [[24, 26], [65, 22], [82, 52], [51, 76], [18, 65]];
const HUB: [[i32; 2]; 5] = [[18, 24], [82, 25], [82, 74], [18, 74], [50, 49]];

fn generate(room: i32, mut districts: [i32; 5], rng: &mut impl Rng) -> MatchLayout {
    districts.shuffle(rng);
    let hub = rng.gen_bool(0.5);
    let anchors = if hub { HUB } else { RING };
    let reflect_x = rng.gen_bool(0.5);
    let reflect_y = rng.gen_bool(0.5);
    let positions = districts
        .iter()
        .zip(anchors)
        .map(|(&id, [x, y])| {
            let x = if reflect_x { 100 - x } else { x };
            let y = if reflect_y { 98 - y } else { y };
            (id, [x + rng.gen_range(-3..=3), y + rng.gen_range(-4..=4)])
        })
        .collect();

    let mut roads = Vec::new();
    let mut optional = Vec::new();
    if hub {
        for i in 0..4 {
            roads.push((4, i));
            optional.push((i, (i + 1) % 4));
        }
    } else {
        for i in 0..5 {
            roads.push((i, (i + 1) % 5));
        }
        let junction = rng.gen_range(0..5);
        optional.extend([
            (junction, (junction + 2) % 5),
            (junction, (junction + 3) % 5),
        ]);
    }
    optional.shuffle(rng);
    // Successive room numbers also differ in road count, even on the same map.
    let road_count = 5 + room.rem_euclid(3) as usize;
    roads.extend(optional.into_iter().take(road_count - roads.len()));
    let mut links: Vec<_> = roads
        .into_iter()
        .map(|(a, b)| {
            let (a, b) = (districts[a], districts[b]);
            [a.min(b), a.max(b)]
        })
        .collect();
    links.sort_unstable();
    MatchLayout { positions, links }
}

pub(super) async fn save(tx: &mut Tx<'_>, room: i32, version: i32) -> Result<(), sqlx::Error> {
    let ids: Vec<i32> =
        sqlx::query_scalar("SELECT id FROM districts WHERE version_id=$1 ORDER BY slot")
            .bind(version)
            .fetch_all(&mut **tx)
            .await?;
    let ids = ids
        .try_into()
        .map_err(|_| sqlx::Error::Protocol("A city match requires five districts".into()))?;
    let layout = generate(room, ids, &mut rand::thread_rng());
    sqlx::query("UPDATE matches SET layout=$1 WHERE id=$2")
        .bind(sqlx::types::Json(layout))
        .bind(room)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use std::collections::HashSet;

    #[test]
    fn rooms_vary_positions_and_roads_and_keep_every_district_reachable() {
        let ids = [47, 28, 92, 16, 31];
        let mut rng = StdRng::seed_from_u64(673);
        let mut networks = HashSet::new();
        let mut positions = HashSet::new();
        for room in 1..=300 {
            let layout = generate(room, ids, &mut rng);
            assert_eq!(layout.links.len(), 5 + (room % 3) as usize);
            assert!(layout.links.windows(2).all(|pair| pair[0] < pair[1]));
            let mut reachable = HashSet::from([ids[0]]);
            for _ in 0..ids.len() {
                for &[a, b] in &layout.links {
                    assert!(a < b && ids.contains(&a) && ids.contains(&b));
                    if reachable.contains(&a) || reachable.contains(&b) {
                        reachable.extend([a, b]);
                    }
                }
            }
            assert_eq!(reachable.len(), ids.len());
            for (i, id) in ids.iter().enumerate() {
                let [x, y] = layout.positions[id];
                assert!((15..=85).contains(&x) && (18..=80).contains(&y));
                for other in &ids[..i] {
                    let [u, v] = layout.positions[other];
                    assert!((x - u).abs() >= 24 || (y - v).abs() >= 20);
                }
            }
            networks.insert(layout.links);
            positions.insert(layout.positions);
        }
        assert!(networks.len() > 50);
        assert_eq!(positions.len(), 300);
    }
}
