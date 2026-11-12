"""Exercise city gameplay against a disposable app and its PostgreSQL database."""
import concurrent.futures
import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/") + "/api/city"


class Client:
    def __init__(self):
        self.http = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def call(self, path, data=None, status=200):
        headers = {"Content-Type": "application/json", "X-City-Request": "1"}
        req = urllib.request.Request(BASE + path, headers=headers,
                                     data=None if data is None else json.dumps(data).encode())
        try:
            response = self.http.open(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read().decode()
            assert response.status == status, (path, status, response.status, body)
            return json.loads(body)


def solve(grid):
    board = list(map(int, grid))
    def fill():
        best = None
        for i, n in enumerate(board):
            if n:
                continue
            r, c = divmod(i, 9)
            box = (r // 3) * 27 + (c // 3) * 3
            used = set(board[r*9:r*9+9] + board[c::9] + [board[box+j*9+k] for j in range(3) for k in range(3)])
            candidates = set(range(1, 10)) - used
            if not candidates:
                return False
            if best is None or len(candidates) < len(best[1]):
                best = (i, candidates)
        if best is None:
            return True
        i, candidates = best
        for n in candidates:
            board[i] = n
            if fill():
                return True
        board[i] = 0
        return False
    assert fill()
    return "".join(map(str, board))


def main():
    admin, a, b, c, d, spectator = [Client() for _ in range(6)]
    for client in [admin, a, b, c, d, spectator]:
        client.call("/session", {})
    a.call("/admin", status=403)
    admin.call("/admin/login", {"key": os.environ["CITY_ADMIN_KEY"]})
    current = a.call("/map")
    start = next(x for x in current["districts"] if x["start"])
    locked = next(x for x in current["districts"] if not x["start"])
    solo = "/campaign/%d/finish" % current["campaign_id"]
    a.call(solo, {"district": locked["id"], "board": solve(locked["givens"])}, 400)
    a.call(solo, {"district": start["id"], "board": solve(start["givens"])})
    assert a.call("/map")["done"] == [start["id"]]
    assert b.call("/map")["done"] == []
    code = admin.call("/matches", {"spectate": True, "minutes": 25, "version": current["version_id"]})["code"]
    room = "/matches/" + code
    a.call(room + "/join", {"side": 0})
    b.call(room + "/join", {"side": 0})
    c.call(room + "/join", {"side": 0}, 409)
    invite = a.call(room)["teams"][1]["invite"]
    c.call(room + "/join", {"invite": invite})
    district = {"district": start["id"]}
    a.call(room + "/play", district, 409)
    d.call(room + "/join", {"side": 1})
    match = a.call(room)
    assert match["status"] == "active" and 1495 <= match["ends"] - match["now"] <= 1500
    assert [len(t["players"]) for t in match["teams"]] == [2, 2]
    spectator.call(room + "/join", {"side": 1}, 409)
    spectator.call(room + "/play", district, 404)
    game = a.call(room + "/play", district)
    assert a.call(room + "/play", district)["id"] == game["id"]
    path = "/games/%d" % game["id"]
    b.call(path + "/finish", {"board": solve(game["givens"])}, 404)
    a.call(path + "/finish", {"board": game["givens"]}, 400)
    a.call(path + "/move", {"board": game["givens"], "revision": 0})
    a.call(path + "/move", {"board": game["givens"], "revision": 0}, 409)
    assert a.call(room + "/play", district)["revision"] == 1
    answer = {"board": solve(game["givens"])}
    with concurrent.futures.ThreadPoolExecutor(2) as workers:
        list(workers.map(lambda _: a.call(path + "/finish", answer), range(2)))
    assert a.call(room)["districts"][0]["scores"] == [1, 0]
    other = c.call(room + "/play", district)
    c.call("/games/%d/finish" % other["id"], {"board": solve(other["givens"])})
    assert a.call(room)["districts"][0]["scores"] == [1, 1]
    b.call(path + "/report", {"reason": "Проверка решения"})
    data = admin.call("/admin")
    report = next(r for r in data["reports"] if r["code"] == code)
    admin.call("/admin/reports/%d" % report["id"], {"withdraw": True, "reason": "Снять задачу"})
    after = a.call(room)
    assert after["districts"][0]["scores"][0] == 0
    assert not next(e for e in after["events"] if e["id"] == game["id"])["enabled"]
    assert not any(e["enabled"] for e in after["events"] if e["puzzle_id"] == report["puzzle_id"])
    ids = [x["id"] for x in current["districts"]]
    layout = dict(name="Контрольная карта", base=current["version_id"], districts=current["districts"],
                  start=0, finish=4, links=[[ids.index(i) for i in link] for link in current["links"]])
    admin.call("/admin/maps", dict(layout, links=[]), 400)
    admin.call("/admin/maps", layout)
    admin.call("/admin/maps", layout, 409)
    assert a.call(room)["map"] == current["name"]
    assert a.call("/map")["version_id"] != current["version_id"]
    assert a.call("/map?campaign=%d" % current["campaign_id"])["done"] == [start["id"]]
    print("Verified 2×2, replay/concurrency, ownership, scoring, moderation, solo and map versions.")


if __name__ == "__main__":
    main()
