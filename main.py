import sys
from pull.Praitepraty import get_praitepraty
from pull.inyourarea import main_inyourarea
from pull.pullpush import mainloop_pullpush
from pull.shinigami_eye.shinigami_eye import main_shinigami_eye
import pull.twitter
from pull.throat import call_get_all_new_rss, call_get_sub_new_rss
# from pull.throat import call_get_all_new_rss, call_get_sub_new_rss, load_throat
print("WARNING: SOFTWARE WILL MAKE COMPUTER GOES BRRRR")
print("RAM WILL GOES BRRRR")
print("CPU WILL GOES BRRRR")
print("GPU WILL GOES BRRRR IF CUDA IS ENABLED ELSE CPU WILL GOES MORE BRRRR")
print("STORAGE WILL GOES BRRRR")
print("INTERNET WILL GOES BRRRR")
print("IF YOU ARE NOT READY, PRESS CTRL+C NOW")
sys.stdout.write('\a')
sys.stdout.write('\a')
sys.stdout.write('\a')
sys.stdout.flush()
# sleep(30)



import asyncio
from pull.ground_news import main_GroundNews
from pull.ifcn import find_ifcn
import pull.pull_html as cr
# from pull.inyourarea import main_inyourarea
from pull.mediabiasfactcheck import main_mediaBiasFactCheck
from pull.reddit import get_subreddits
from pull.wikidata import  get_temp_data_wikidata
import pull.twitter
import pull.mastodon_helper as mastodon
from utility.helper import do_callbacks, helper_setup
from utility.task import add_task, run_tasks, task_init


async def main():
    await helper_setup()
    # utility.nlp.init_nlp()
    await mastodon.mastodon_main()
    # load_throat()
    # reddit_init()
    await task_init()
    await add_task(get_temp_data_wikidata)
    await add_task(find_ifcn)
    await add_task(get_praitepraty)
    await add_task(main_mediaBiasFactCheck)
    await add_task(main_GroundNews)
    await add_task(main_inyourarea)
    # await add_task(call_get_all_new_rss)
    # await add_task(call_get_sub_new_rss)
    # await add_task(get_subreddits)
    # await add_task(mastodon.mastodon_bot_get_live_local_feed)
    # await add_task(mastodon.mastodon_bot_get_live_globe_feed)
    await add_task(mainloop_pullpush)
    print("running tasks")
    ccc = asyncio.create_task(do_callbacks())
    await run_tasks()
    await ccc
    
asyncio.run(main())
