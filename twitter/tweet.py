from typing import Optional
import logging
import time
import psycopg
import requests
import dataclasses
import TwitterAPI as t
import utils

logging.getLogger().setLevel(logging.INFO)

twitter_api = t.TwitterAPI(
    *utils.get_secret("TWITTER_CREDS").split(","),
    api_version="1.1",
)

view_url = "https://mcltajcadcrkywecsigc.supabase.in/storage/v1/object/public/imoges/{slug}.png"


admin_signal_url = "https://imogen-renaissance.fly.dev"


def admin(msg: str) -> None:
    """send a message to admin"""
    logging.info(msg)
    requests.post(
        f"{admin_signal_url}/admin",
        params={"message": str(msg)},
    )


@dataclasses.dataclass
class Prompt:
    "holds database result with prompt information"
    prompt: str
    filepath: str

# where not params::jsonb ? 'nopost'

def get_prompt() -> tuple[Optional[str], Optional[str]]:
    """Gets the fairiest prompt of them all, the one who got the most reacts form all the land"""
    conn = psycopg.connect(utils.get_secret("DATABASE_URL"), autocommit=True)
    ret = conn.execute(
        """select prompt, filepath from prompt_queue where now() - inserted_ts < '1 hour'
        and not params::jsonb ? 'nopost'
        and filepath is not null
        order by map_len(reaction_map) desc, loss asc limit 1;"""
    ).fetchone()
    logging.info(ret)
    if not ret:
        return None, None
    prompt, filepath = ret
    slug = (
        filepath.removeprefix("output/").removesuffix(".png").removesuffix("/progress")
    )
    return prompt, view_url.format(slug=slug)


def post_tweet(prompt: str, url: str) -> None:
    "post tweet, either all at once for images or in chunks for videos"
    logging.info("uploading to twitter")
    picture = requests.get(url).content
    media_resp = twitter_api.request(
        "media/upload", None, {"media": picture, "media_type": "image/png"}
    )
    # if not prompt.filepath.endswith("mp4"):
    # else:
    # bytes_sent = 0
    # total_bytes = os.path.getsize(prompt.filepath)
    # file = open(prompt.filepath, "rb")
    # init_req = twitter_api.request(
    #     "media/upload",
    #     {"command": "INIT", "media_type": "video/mp4", "total_bytes": total_bytes},
    # )

    # media_id = init_req.json()["media_id"]
    # segment_id = 0

    # while bytes_sent < total_bytes:
    #     chunk = file.read(4 * 1024 * 1024)
    #     twitter_api.request(
    #         "media/upload",
    #         {
    #             "command": "APPEND",
    #             "media_id": media_id,
    #             "segment_index": segment_id,
    #         },
    #         {"media": chunk},
    #     )
    #     segment_id = segment_id + 1
    #     bytes_sent = file.tell()
    # logging.info("media was video. Ignoring for now")
    # media_resp = twitter_api.request(
    #     "media/upload", {"command": "FINALIZE", "media_id": media_id}
    # )
    try:
        logging.info(media_resp)
        media = media_resp.json()
        media_id = media["media_id"]
        twitter_post = {
            "status": prompt,
            "media_ids": media_id,
        }
        twitter_api.request("statuses/update", twitter_post)
    except KeyError:
        try:
            logging.error(media_resp.text)
            admin(media_resp.text)
        except:  # pylint: disable=bare-except
            logging.error("couldn't send to admin")


if __name__ == "__main__":
    while True:
        prompt, view_url = get_prompt()
        if view_url and prompt:
            post_tweet(prompt, view_url)
        time.sleep(60 * 60)
