import aiohttp
import pghelp

queue = pghelp.Queue

async def submit_prompt(text, params, user/user context, ...):
    ...
    Queue.enqueue_any(...)
    ensure_worker(...)
    return id, url, queue position 

async def get_prompt

# client:

# /prompt# PUT
# 302 -> /prompt/id?raw=1

# /prompt/id
# DELETE

# /prompt/id
# GET
# either graphical ui or json of 
# {
# queue position
# predicted wait time
# id
# }

# /prompt/wss
# websocket experience:
#     - you get a message with a cdn to the image
#     - you can directly get the image or a stream over the socket

# ---
# worker:

# /workers/prompt
# GET
# X-Worker-Hostname: ...
# X-Worker-Tag: ...
# either no prompts or 302 to /prompt/id, possibly also immediately claiming it

# /prompt/<id>/worker
# PATCH
# claim the image to work on

# /prompt/<id>/worker
# POST
# upload the image

# /prompt/<id>/metadata
# PATCH
# update e.g. reactions, likes, twitter url, permalink)
