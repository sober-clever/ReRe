# based on the implementation of D3, adapted for the Yelp Open Dataset
import fire
from loguru import logger
import json
from tqdm import tqdm
import random
import calendar
import datetime
import csv
import os
from collections import defaultdict, Counter


def get_timestamp_start(year, month, day=1):
    # NOTE: unlike process.py we use timegm (UTC) instead of datetime.timestamp(),
    # which silently depends on the machine's local timezone.
    return calendar.timegm(datetime.datetime(year=year, month=month, day=day).timetuple())


def parse_date(s):
    return calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())


def load_business(path):
    """business_id -> {name, city, state, addr}, keeping only usable names."""
    meta = {}
    dropped = 0
    with open(path) as f:
        for line in tqdm(f, desc="business"):
            d = json.loads(line)
            name = d.get("name") or ""
            name = name.replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
            # same title filter as process.py: drop empty and overly long names
            if len(name) > 1 and len(name.split(" ")) <= 20:
                meta[d["business_id"]] = {
                    "name": name,
                    "city": (d.get("city") or "").strip(),
                    "state": (d.get("state") or "").strip(),
                    "addr": (d.get("address") or "").strip(),
                }
            else:
                dropped += 1
    logger.info(f"business: kept {len(meta)}, dropped {dropped} (bad/long name)")
    return meta


def load_reviews(path, meta, start_timestamp, end_timestamp):
    """Stream the 5GB review file, keeping only in-window reviews of known businesses."""
    reviews = []
    total = 0
    with open(path) as f:
        for line in tqdm(f, desc="review"):
            total += 1
            # cheap substring pre-filter to avoid json-parsing all 7M lines
            i = line.find('"date":"')
            if i > -1 and not (start_timestamp <= parse_date(line[i + 8:i + 27]) <= end_timestamp):
                continue
            d = json.loads(line)
            bid = d["business_id"]
            if bid not in meta:
                continue
            ts = parse_date(d["date"])
            if ts < start_timestamp or ts > end_timestamp:
                continue
            reviews.append((d["user_id"], bid, d["stars"], ts))
    logger.info(f"review: scanned {total}, kept {len(reviews)} in window")
    return reviews


def k_core(reviews, K):
    """Iteratively drop users/items below K until a fixed point (process.py:66-102)."""
    remove_users, remove_items = set(), set()
    it = 0
    while True:
        it += 1
        users, items = defaultdict(int), defaultdict(int)
        kept = []
        for r in reviews:
            if r[0] in remove_users or r[1] in remove_items:
                continue
            users[r[0]] += 1
            items[r[1]] += 1
            kept.append(r)
        flag = False
        for u in users:
            if users[u] < K:
                remove_users.add(u)
                flag = True
        for i in items:
            if items[i] < K:
                remove_items.add(i)
                flag = True
        density = len(kept) / (len(users) * len(items)) if users and items else 0
        logger.info(f"  iter {it}: users={len(users)} items={len(items)} "
                    f"interactions={len(kept)} density={density * 100:.4f}%")
        if not flag:
            break
    assert min(users.values()) >= K and min(items.values()) >= K, "k-core not satisfied"
    logger.info(f"{K}-core converged after {it} iterations")
    return kept, users, items


def join_parts(parts):
    """Join location components, dropping empties and order-preserving dedupe.

    Yelp's `address` is sometimes blank (delivery-only brands) or repeats itself
    ("1370 Big Fish Dr, Legends at Sparks, Legends at Sparks"), which would
    otherwise produce a stray comma or a stuttering title.
    """
    out = []
    for p in parts:
        p = p.strip()
        if p and p not in out:
            out.append(p)
    return ", ".join(out)


def disambiguate_titles(item_ids, meta, title_level="city"):
    """Append a location suffix to titles shared by several businesses.

    Restaurant chains share a name (34x "First Watch"), and calc.py scores a hit
    by string equality, so a bare chain name would credit a prediction for the
    wrong branch.

    title_level="city": suffix is (City, ST). Short, natural titles, but ~225
        names with several branches in one city stay ambiguous (~5% of test rows).
    title_level="addr": same, then escalates only the still-colliding items to
        (Address, City, ST). Fully unique. Escalating on demand rather than
        addressing every duplicate keeps ~1000 titles short, since two thirds of
        the duplicated names are already unique once the city is known.
    title_level="always_addr": every title carries (Address, City, ST), whether
        or not the name is ambiguous. This is what the published Yelp dataset
        does -- titles average 9.3 words instead of 3.3, but are unique by
        construction.
    """
    assert title_level in ("city", "addr", "always_addr"), \
        f"bad title_level: {title_level}"
    counts = Counter(meta[b]["name"] for b in item_ids)
    dupes = {n for n, c in counts.items() if c > 1}
    logger.info(f"duplicated base names: {len(dupes)} "
                f"({sum(counts[n] for n in dupes)} items affected)")

    def city_suffix(m):
        return f"{m['name']} ({join_parts([m['city'], m['state']])})"

    def addr_suffix(m):
        return f"{m['name']} ({join_parts(m['addr'].split(',') + [m['city'], m['state']])})"

    if title_level == "always_addr":
        title = {b: addr_suffix(meta[b]) for b in item_ids}
        residual = {t for t, c in Counter(title.values()).items() if c > 1}
        if residual:
            seen = defaultdict(int)
            for b in item_ids:
                if title[b] in residual:
                    seen[title[b]] += 1
                    title[b] = f"{title[b]} #{seen[title[b]]}"
        distinct = len(set(title.values()))
        logger.info(f"title_level=always_addr: {distinct} distinct titles for "
                    f"{len(item_ids)} items")
        assert distinct == len(item_ids), "titles are not unique"
        longest = max(title.values(), key=lambda t: len(t.split(" ")))
        words = [len(t.split(" ")) for t in title.values()]
        logger.info(f"title words mean={sum(words)/len(words):.2f} "
                    f"max={len(longest.split(' '))}")
        return title

    title = {}
    for b in item_ids:
        m = meta[b]
        title[b] = m["name"] if m["name"] not in dupes else city_suffix(m)

    if title_level == "addr":
        collide = {t for t, c in Counter(title.values()).items() if c > 1}
        escalated = 0
        for b in item_ids:
            if title[b] in collide:
                title[b] = addr_suffix(meta[b])
                escalated += 1
        logger.info(f"escalated {escalated} items to (Address, City, ST)")

        # same name, same city, same address: fall back to an index
        residual = {t for t, c in Counter(title.values()).items() if c > 1}
        if residual:
            logger.warning(f"{len(residual)} titles collide even with address; "
                           f"appending an index")
            seen = defaultdict(int)
            for b in item_ids:
                if title[b] in residual:
                    seen[title[b]] += 1
                    title[b] = f"{title[b]} #{seen[title[b]]}"

    distinct = len(set(title.values()))
    logger.info(f"title_level={title_level}: {distinct} distinct titles for "
                f"{len(item_ids)} items ({len(item_ids) - distinct} share a title)")
    if title_level == "addr":
        assert distinct == len(item_ids), "titles are not unique"
    longest = max(title.values(), key=lambda t: len(t.split(" ")))
    assert len(longest.split(" ")) <= 20, f"title too long: {longest}"
    logger.info(f"longest title is {len(longest.split(' '))} words")
    return title


def merge_same_name(kept, meta):
    """Collapse businesses sharing (name, city, state) into one item.

    Yelp's unit is a storefront, so a chain has one record per branch (6x
    "French Truck Coffee" in New Orleans). Since calc.py scores a hit by string
    equality, those branches are indistinguishable at eval time; merging them
    makes the item set match what the titles can actually express.

    Runs AFTER k_core, on the survivors only. Merging first would instead let
    sub-threshold branches pool their interactions past the 5-core cut, which
    cascades into more surviving users and *raises* the item count (8824 ->
    9307). Merging afterwards leaves users and interactions untouched, and
    cannot break the 5-core property: a merged item's count is the sum of
    counts that were each already >= K.
    """
    groups = defaultdict(list)
    for bid in {r[1] for r in kept}:
        m = meta[bid]
        groups[(m["name"], m["city"], m["state"])].append(bid)

    canon = {}
    for key, bids in groups.items():
        rep = min(bids)  # deterministic representative
        for b in bids:
            canon[b] = rep

    collapsed = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    n_groups = sum(1 for v in groups.values() if len(v) > 1)
    logger.info(f"merged same (name, city, state): {n_groups} groups absorbed "
                f"{collapsed} duplicate items -> {len(groups)} items remain")
    return [(u, canon[b], s, t) for u, b, s, t in kept]


def gao(
    category="Yelp",
    business_file="../Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json",
    review_file="../Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json",
    output_dir="./Amazon",
    K=5,
    st_year=2021, st_month=1, st_day=1,
    ed_year=2022, ed_month=1, ed_day=1,
    max_len=10,
    stamp=None,
    title_level="city",
    merge_same_name_items=False,
    shuffle_items=True,
):
    start_timestamp = get_timestamp_start(st_year, st_month, st_day)
    end_timestamp = get_timestamp_start(ed_year, ed_month, ed_day)
    logger.info(f"window: {start_timestamp} .. {end_timestamp} "
                f"({st_year}-{st_month}-{st_day} .. {ed_year}-{ed_month}-{ed_day})")

    meta = load_business(business_file)
    reviews = load_reviews(review_file, meta, start_timestamp, end_timestamp)

    kept, users, items = k_core(reviews, K)

    if merge_same_name_items:
        kept = merge_same_name(kept, meta)
        items = defaultdict(int)
        for r in kept:
            items[r[1]] += 1
        assert min(items.values()) >= K, "merge broke the k-core property"

    item_list = list(items.keys())
    title = disambiguate_titles(item_list, meta, title_level=title_level)
    if merge_same_name_items:
        assert len(set(title.values())) == len(item_list), \
            "merged items should have unique titles"

    # shuffle items and assign ids (process.py:113-124)
    if shuffle_items:
        random.seed(42)
        random.shuffle(item_list)
    item2id = {}
    if stamp is None:
        stamp = f"{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}"
    info_dir = os.path.join(output_dir, "info")
    os.makedirs(info_dir, exist_ok=True)
    with open(os.path.join(info_dir, f"{stamp}.txt"), "w") as f:
        for count, b in enumerate(item_list):
            item2id[b] = count
            f.write(f"{title[b]}\t{count}\n")

    # group per user, order chronologically
    interact = defaultdict(lambda: {"items": [], "ratings": [], "timestamps": []})
    for user, bid, stars, ts in kept:
        interact[user]["items"].append(bid)
        interact[user]["ratings"].append(stars)
        interact[user]["timestamps"].append(ts)

    interaction_list = []
    for key in tqdm(interact.keys(), desc="sequences"):
        rec = interact[key]
        res = sorted(zip(rec["items"], rec["ratings"], rec["timestamps"]), key=lambda x: int(x[2]))
        bids, ratings, timestamps = (list(x) for x in zip(*res))
        item_ids = [item2id[b] for b in bids]
        titles = [title[b] for b in bids]
        for i in range(1, len(bids)):
            st = max(i - max_len, 0)
            interaction_list.append([
                key,
                bids[st:i], bids[i],
                item_ids[st:i], item_ids[i],
                titles[st:i], titles[i],
                ratings[st:i], ratings[i],
                timestamps[st:i], timestamps[i],
            ])
    logger.info(f"interaction_list: {len(interaction_list)}")

    # chronological 8:1:1 split on the target timestamp (process.py:167-186)
    interaction_list = sorted(interaction_list, key=lambda x: int(x[-1]))
    n = len(interaction_list)
    header = ['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id',
              'history_item_title', 'item_title', 'history_rating', 'rating',
              'history_timestamp', 'timestamp']
    splits = {
        "train": interaction_list[:int(n * 0.8)],
        "valid": interaction_list[int(n * 0.8):int(n * 0.9)],
        "test": interaction_list[int(n * 0.9):],
    }
    for name, rows in splits.items():
        d = os.path.join(output_dir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{stamp}.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        logger.info(f"{name.capitalize()} {category}: {len(rows)}")

    n_u, n_i = len(users), len(item_list)
    logger.info("=" * 60)
    logger.info(f"Users={n_u}  Items={n_i}  Interactions={n}  "
                f"Density={(n + n_u) / (n_u * n_i) * 100:.3f}%")
    logger.info(f"Train={len(splits['train'])} Valid={len(splits['valid'])} "
                f"Test={len(splits['test'])}")
    logger.info("Done!")


if __name__ == '__main__':
    fire.Fire(gao)
