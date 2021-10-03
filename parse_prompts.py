import imagegen

import json
import sys
from pathlib import Path
def parse_file_once(fname="prompts.json"):
    with open(fname) as f:
        prompts = json.load(f)
    try:
        current_prompt = prompts.pop(0)
    except IndexError:
        return False
    current_args = base_args.with_update(current_prompt)

    name = (
        current_prompt.get("init_image")
        or current_prompt["text"].replace(" ", "_")
        or "no_name"
    )
    no_punctuation = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "-", no_punctuation).strip("-_")
    directory = Path("./results") / slug
    try:
        os.mkdir(directory)
    except FileExistsError:
        print("dir already exists")
    generate(current_args)
    try:
        for file in Path("steps").glob("*"):
            shutil.move(file, directory)
    except FileNotFoundError:
        print("file not found error moving")
    try:
        finished = json.load(open("finished_prompts.json")) + [current_prompt]
    except FileNotFoundError:
        finished = [current_prompt]
    json.dump(finished, open("finished_prompts.json", "w"), indent=4)
    json.dump(prompts, open(fname, "w"), indent=4)

    return True


if __name__ == "__main__":
    while parse_file_once():
        print("parsing next file")
#    parse_file_once()
