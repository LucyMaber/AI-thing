import requests

def find_feeds(url):
    cdx_url = f'http://web.archive.org/cdx/search/cdx?url={url}/*&output=json&filter=statuscode:200&mimetype:text/xml'
    response = requests.get(cdx_url)
    if response.status_code == 200:
        results = response.json()
        feeds = []
        for result in results[1:]:
            feeds.append(result[2])
        return feeds
    else:
        print("Error:", response.status_code)


