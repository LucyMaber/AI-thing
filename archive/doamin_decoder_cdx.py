import os
import sys
from urllib.parse import urlparse
from pathlib import Path

from pull.feedfinder import pass_feed_viewer
from pull.reddit import scrape_listing

url_pattons = [
    "libredd.it/user",
    "teddit.net/u",
    "unavatar.now.sh/reddit",
    "new.reddit.com/user",
    "reddit.com/u",
    "old.reddit.com/u",
    "old.reddit.com/user",
    "reddit.com/u",
    "reddit.com/user",
    "ovarit.com/o",
    "ovarit.com/u",
    "tiktok.com/place",
    "tiktok.com/music",
    "flickr.com/photos",
    "flickr.com/people",
    "secure.flickr.com/photos",
    "secure.flickr.com/people",
    "instagram.com",
    "instagram.com/_u",
    "threads.net",
    "unavatar.now.sh/instagram",
]


subreddits = {}
pages = {}
users = {}
posts = {}
domain_list = set()
user_list = set()


def get_domains():
    return domain_list


def add_domain(domain):
    if domain not in domain_list:
        domain_list.add(domain)


def resource_domain(domain, data):
    if domain in domain_list:
        pass_feed_viewer(data)


def resource_url(domain, url, mime, data):
    if domain in domain_list:
        pass


async def helper_cache_url(url, dt_object, data, where, location=None, mime=None):
    try:
        url_ = urlparse(url)
        p = Path(url_.path)
        if len(p.parts) < 3:
            return None
        if "libredd.it/r/" in url:
            await add_resource_reddit(
                "subreddit", p.parts[2], dt_object, where, data, location=None)
            pass
        if "redditgrid.com/r/" in url:
            await add_resource_reddit(
                "subreddit", p.parts[2], dt_object,  where, data, location=location)
            pass
        if "teddit.net/r/" in url:
            await add_resource_reddit(
                "subreddit", p.parts[2], dt_object,  where, data, location=location)
        if "subredditstats.com/r/" in url:
            await add_resource_reddit(
                "subreddit", p.parts[2], dt_object,  where, data, location=location)
        if "removeddit.com/r/" in url:
            await add_resource_reddit(
                "subreddit", p.parts[2], dt_object,  where, data, location=location)
            pass
        if "reddit.com/r/" in url:
            await add_resource_reddit(
                "subreddit", p.parts[2], dt_object, where, data, location=None)
            if len(p.parts) > 4 and p.parts[3] == "comments":
                post_ = p.parts[2]+"/comments/"+p.parts[4]
                await add_resource_reddit("post", post_, dt_object,
                                          where, data, location=location)
            else:
                if ("html" in mime) and (data is not None):
                    await scrape_listing(data)
        if "libredd.it/user/" in url:
            await add_resource_reddit(
                "user", p.parts[2], dt_object,  where, data, location=location)
            pass
        if "teddit.net/u/" not in url:
            await add_resource_reddit(
                "user", p.parts[2], dt_object,  where, data, location=location)
        if "unavatar.now.sh/reddit/" in url:
            await add_resource_reddit(
                "subreddit", p.parts[2], dt_object,  where, data, location=location)
        if "reddit.com/user/" in url:
            await add_resource_reddit(
                "user", p.parts[2], dt_object,  where, data, location=location)
            pass
        if "reddit.com/u/" in url:
            await add_resource_reddit(
                "user", p.parts[2], dt_object,  where, data, location=location)
            pass
        if "reddit.com/domain/" in url:
            await add_resource_reddit(
                "domain", p.parts[2], dt_object,  where, data, location=location)
            resource_domain(url_.hostname, data)
            resource_url(url_.hostname, url, mime, data)
            if "html" in mime:
                await scrape_listing(data)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("parts:", p.parts, len(p.parts))
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("exc_type",exc_type, fname, exc_tb.tb_lineno)
        print("helper_cache_url:", e)


async def add_resource_reddit(property, value, time, where, data, location=None):
    try:
        if "subreddit" == property:
            if value not in subreddits.keys():
                subreddits[value] = {
                    "value": value,
                    "start": {"time": time, "where": where, "location": location},
                    "end": {"time": time, "where": where, "location": location}
                }
            else:
                if subreddits[value]["end"]["time"] < time:
                    subreddits[value]["end"]["time"] = time
                    subreddits[value]["end"]["where"] = where
                    subreddits[value]["start"]["location"] = location
                if subreddits[value]["start"]["time"] > time:
                    subreddits[value]["start"]["time"] = time
                    subreddits[value]["start"]["where"] = where
                    subreddits[value]["start"]["location"] = location
        if "page" == property:
            if value not in pages.keys():
                pages[value] = {
                    "start": {"time": time, "where": where, "location": location},
                    "end": {"time": time, "where": where, "location": location}
                }
            else:
                if pages[value]["end"]["time"] < time:
                    pages[value]["end"]["time"] = time
                    pages[value]["end"]["where"] = where
                    pages[value]["start"]["location"] = location
                if pages[value]["start"]["time"] > time:
                    pages[value]["start"]["time"] = time
                    pages[value]["start"]["where"] = where
                    pages[value]["start"]["location"] = location
        if "user" == property:
            if value not in users.keys():
                users[value] = {
                    "value": value,
                    "start": {"time": time, "where": where, "location": location},
                    "end": {"time": time, "where": where, "location": location}
                }
            else:
                if users[value]["end"]["time"] < time:
                    users[value]["end"]["time"] = time
                    users[value]["end"]["where"] = where
                    users[value]["start"]["location"] = location
                if users[value]["start"]["time"] > time:
                    users[value]["start"]["time"] = time
                    users[value]["start"]["where"] = where
                    users[value]["start"]["location"] = location
        if "post" == property:
            if value not in posts.keys():
                posts[value] = {
                    "start": {"time": time, "where": where, "location": location},
                    "end": {"time": time, "where": where, "location": location}
                }
            else:
                if posts[value]["end"]["time"] < time:
                    posts[value]["end"]["time"] = time
                    posts[value]["end"]["where"] = where
                    posts[value]["start"]["location"] = location
                if posts[value]["start"]["time"] > time:
                    posts[value]["start"]["time"] = time
                    posts[value]["start"]["where"] = where
                    posts[value]["start"]["location"] = location
        if "domain" == property:
            if value not in domain_list:
                pass
            else:
                pass
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("add_resource_reddit:", e)
