import json
import TwitterAPI as t
import main_ganclip_hacking as clipart

api = t.TwitterAPI(
    "qxmCL5ebziSwOIlf3MByuhRvY",
    "3sj1HeUXPeZ3YEG45j1fa1ckGvCQI2lTmg39QUue1bK69KPtGL",
    "1442633760315375621-UreMIwMZK3x7Povds8A4ruEbS7VPeD",
    "INQ5JoET33lxjoIyT8VO557iPFd9Y2uAuxhZbUUeepzQq",
    api_version="1.1",
)
username = "@dreambs3"
api.request("statuses/update", {"status": "[lively beep boop noises"])
stream = api.request("statuses/filter", {"track": username})
print(stream)
try: 
    for item in stream:
        print(item)
        user = item.get("user", {}).get("screen_name")
        status_id = item.get("id")
        text = item.get("text")
        if user and status_id and text:
            args = clipart.base_args.with_update(
                {"text": text.removeprefix(username).strip()}
            )
            clipart.generate(args)
            f = open("progress.png", mode="rb").read()
            media = api.request("media/upload", None, {"media": f}).json()
            media_id = media["media_id"]
            post = {
                "status": "@" + user,
                "in_reply_to_status_id": status_id,
                "media_ids": media_id,
            }
            req = api.request("statuses/update", post)
            try:
                print(req.json())
            except json.JSONDecodeError:
                print(req.text)

except KeyboardInterrupt:
    api.request("statuses/update", {"status": "[sleepy beep boop noises"])

