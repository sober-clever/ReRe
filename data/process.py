# based on the implementation of D3
import fire
from loguru import logger
import json
from tqdm import tqdm
import random
import time
import datetime
import csv
import os
    

def get_timestamp_start(year, month):
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())

def gao(category, metadata=None, reviews=None, K=5, st_year=2017, st_month=10, ed_year=2018, ed_month=11, output=True):
    if st_year < 1996:
        return
    start_timestamp = get_timestamp_start(st_year, st_month)
    end_timestamp = get_timestamp_start(ed_year, ed_month)
    logger.info(f"from {start_timestamp} to {end_timestamp}")
    if metadata is None:
        with open(f'../meta_{category}.json') as f:
            # print(1)
            metadata = [json.loads(line) for line in f]
        try:
            with open(f'../{category}_5.json') as f:
                reviews = [json.loads(line) for line in f]
        except:
            with open(f'../{category}.json') as f:
                reviews = [json.loads(line) for line in f]
    else:
        metadata = metadata
        reviews = reviews
    logger.info(f"from {category} metadata: {len(metadata)} reviews: {len(reviews)}")
    
    # item: asin, title, price, imUrl, related, salesRank, categories, description
    users = set()
    items = set()
    for review in tqdm(reviews):
        if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
            continue        
        users.add(review['reviewerID'])
        items.add(review['asin'])
    # print(len(users), len(items))

    logger.info(f"users: {len(users)}, items: {len(items)}, reviews: {len(reviews)}, density: {len(reviews) / (len(users) * len(items))}")
    remove_users = set()
    remove_items = set()
    
    
    id_title = {}
    for meta in tqdm(metadata):
        if ('title' not in meta) or (meta['title'].find('<span id') > -1):
            remove_items.add(meta['asin'])
            continue
        meta['title'] = meta["title"].replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
        if len(meta['title']) > 1 and len(meta['title'].split(" ")) <= 20: # remove the item without title # remove too long title
            id_title[meta['asin']] = meta['title']
        else:
            remove_items.add(meta['asin'])
    for review in tqdm(reviews):
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])

    while True:
        new_reviews = []
        flag = False
        total = 0
        users = dict()
        items = dict()
        new_reviews = []
        for review in tqdm(reviews):
            if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                continue
            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue          
            if review['reviewerID'] not in users:
                users[review['reviewerID']] = 0
            users[review['reviewerID']] += 1
            if review['asin'] not in items:
                items[review['asin']] = 0
                
            items[review['asin']] += 1
            total += 1
            new_reviews.append(review)
            
        for user in users:
            if users[user] < K:
                remove_users.add(user)
                flag = True
        
        for item in items:
            if items[item] < K:
                remove_items.add(item)
                flag = True

        logger.info(f"users: {len(users)}, items: {len(items)}, reviews: {total}, density: {total / (len(users) * len(items))}")
        if st_year > 1996 and len(items) < 3000:
            break
        if not flag:
            break

    if st_year > 1996 and len(items) < 3000:
        gao(category, metadata = metadata, reviews = reviews, K=K, st_year=st_year - 1, st_month=st_month, ed_year=ed_year, ed_month=ed_month, output=True)
        return
        

    logger.info(f"remove_users: {len(remove_users)}, remove_items: {len(remove_items)}")
    
    reviews = new_reviews
    # shuffle items and assign the id to each item
    items = list(items.keys())
    random.seed(42)
    random.shuffle(items)
    item2id = dict()
    count = 0
    if not os.path.exists("./info"):
        os.mkdir("./info")
    with open(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}.txt", 'w') as f:
        for item in items:
            item2id[item] = count
            f.write(f"{id_title[item]}\t{count}\n")
            count += 1
            
    if not output:
        return 
    # get all the review data

    interact = dict()
    old_items = set(items)
    new_items = set()
    for review in tqdm(new_reviews):
        user = review['reviewerID']
        item = review['asin']
        if user not in interact:
            interact[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
                'reviews': []
            }
        new_items.add(item)
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['overall'])
        interact[user]['timestamps'].append(review['unixReviewTime'])
    interaction_list = []   
    for key in tqdm(interact.keys()):
        items = interact[key]['items']
        ratings = interact[key]['ratings']
        timestamps = interact[key]['timestamps']
        all = list(zip(items, ratings, timestamps))
        res = sorted(all, key=lambda x: int(x[2]))
        items, ratings, timestamps = zip(*res)
        items, ratings, timestamps = list(items), list(ratings), list(timestamps)
        interact[key]['items'] = items
        interact[key]['ratings'] = ratings
        interact[key]['timestamps'] = timestamps
        interact[key]['item_ids'] = [item2id[item] for item in items]
        interact[key]['title'] = [id_title[item] for item in items]
        for i in range(1, len(items)):
            st = max(i - 10, 0)
            interaction_list.append([key, interact[key]['items'][st:i], interact[key]['items'][i], interact[key]['item_ids'][st:i], interact[key]['item_ids'][i], interact[key]['title'][st:i], interact[key]['title'][i], interact[key]['ratings'][st:i], interact[key]['ratings'][i], interact[key]['timestamps'][st:i], interact[key]['timestamps'][i]])
    logger.info(f"interaction_list: {len(interaction_list)}")

    # split train valid test
    interaction_list = sorted(interaction_list, key=lambda x: int(x[-1]))
    # check if the train valid test file exits, if not create
    if not os.path.exists("./train"):
        os.mkdir("./train")
    if not os.path.exists("./valid"):
        os.mkdir("./valid")
    if not os.path.exists("./test"):
        os.mkdir("./test")
    with open(f"./train/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'history_timestamp', 'timestamp'])
        writer.writerows(interaction_list[:int(len(interaction_list) * 0.8)])
    with open(f"./valid/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'history_timestamp', 'timestamp'])
        writer.writerows(interaction_list[int(len(interaction_list) * 0.8):int(len(interaction_list) * 0.9)])
    with open(f"./test/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'history_timestamp', 'timestamp'])
        writer.writerows(interaction_list[int(len(interaction_list) * 0.9):])
    
    logger.info(f"Train {category}: {len(interaction_list[:int(len(interaction_list) * 0.8)])}")
    logger.info(f"Valid {category}: {len(interaction_list[int(len(interaction_list) * 0.8):int(len(interaction_list) * 0.9)])}")
    logger.info(f"Test {category}: {len(interaction_list[int(len(interaction_list) * 0.9):])}")
    logger.info("Done!")



    

if __name__ == '__main__':
    fire.Fire(gao)