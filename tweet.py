import TwitterAPI as t
import psycopg
import requests
from config import get_secret

twitter_api = t.TwitterAPI(
    *utils.get_secret("TWITTER_CREDS").split(","),
    api_version="1.1",
)

view_url = "https://mcltajcadcrkywecsigc.supabase.in/storage/v1/object/public/imoges/{slug}.png"

conn = psycopg.connect(get_secret("DATABASE_URL"), autocommit=True)

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

def get_prompt() -> tuple[Prompt, str]:
    ret = conn.fetch(
        """select prompt, filepath from prompt_queue where now() - inserted_ts < '1 hour'
        order by map_len(reaction_map) desc, loss asc limit 1;"""
    )
    logging.info(ret)
    if not ret or not (filepath := ret[0].get("filepath")):
        return "sorry, I don't have that image saved for upsampling right now"
    prompt = Prompt(**ret[0])
    # adjust for mp4s
    slug = (
        filepath.removeprefix("output/").removesuffix(".png").removesuffix("/progress")
    )
    return prompt, view_url.format(slug)

def post_tweet(prompt: Prompt, url: str) -> None:
    "post tweet, either all at once for images or in chunks for videos"
    logging.info("uploading to twitter")
    requests.get(url)
    if not prompt.filepath.endswith("mp4"):
        media_resp = twitter_api.request(
            "media/upload", None, {"media": open(prompt.filepath, mode="rb").read()}
        )
    else:
        bytes_sent = 0
        total_bytes = os.path.getsize(prompt.filepath)
        file = open(prompt.filepath, "rb")
        init_req = twitter_api.request(
            "media/upload",
            {"command": "INIT", "media_type": "video/mp4", "total_bytes": total_bytes},
        )

        media_id = init_req.json()["media_id"]
        segment_id = 0

        while bytes_sent < total_bytes:
            chunk = file.read(4 * 1024 * 1024)
            twitter_api.request(
                "media/upload",
                {
                    "command": "APPEND",
                    "media_id": media_id,
                    "segment_index": segment_id,
                },
                {"media": chunk},
            )
            segment_id = segment_id + 1
            bytes_sent = file.tell()
        media_resp = twitter_api.request(
            "media/upload", {"command": "FINALIZE", "media_id": media_id}
        )
    try:
        media = media_resp.json()
        media_id = media["media_id"]
        twitter_post = {
            "status": prompt.prompt,
            "media_ids": media_id,
        }
        twitter_api.request("statuses/update", twitter_post)
    except KeyError:
        try:
            logging.error(media_resp.text)
            admin(media_resp.text)
        except:  # pylint: disable=bare-except
            logging.error("couldn't send to admin")

if __name__=="__main__":
    pass
