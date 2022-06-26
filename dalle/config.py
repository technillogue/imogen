# Copyright (c) 2022 Forest Contact
import os
import functools
import logging
from typing import Optional, cast

# import logging
# handler = logging.FileHandler("debug.log")
# handler.setLevel("DEBUG")
# logging.getLogger().addHandler(handler)
# fmt = logging.Formatter("{levelname} {module}:{lineno}: {message}", style="{")
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(fmt)
# handler.setLevel("DEBUG")
# logging.getLogger().addHandler(stream_handler)
# logging.info("starting")
# logging.debug("debug")
# tee = subprocess.Popen(["tee", "-a", "fulllog.txt"], stdin=subprocess.PIPE)
# # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
# # of any child processes we spawn)
# os.dup2(tee.stdin.fileno(), sys.stdout.fileno())  # type: ignore
# os.dup2(tee.stdin.fileno(), sys.stderr.fileno())  # type: ignore

@functools.cache  # don't load the same env more than once
def load_secrets(env: Optional[str] = None, overwrite: bool = False) -> None:
    if not env:
        env = os.environ.get("ENV", "dev")
    try:
        logging.info("loading secrets from %s_secrets", env)
        secrets = [
            line.strip().split("=", 1)
            for line in open(f"{env}_secrets")
            if line and not line.startswith("#")
        ]
        can_be_a_dict = cast(list[tuple[str, str]], secrets)
        if overwrite:
            new_env = dict(can_be_a_dict)
        else:
            new_env = (
                dict(can_be_a_dict) | os.environ
            )  # mask loaded secrets with existing env
        os.environ.update(new_env)
    except FileNotFoundError:
        pass


def get_secret(key: str, env: Optional[str] = None) -> str:
    try:
        secret = os.environ[key]
    except KeyError:
        load_secrets(env)
        secret = os.environ.get(key) or ""  # fixme
    if secret.lower() in ("0", "false", "no"):
        return ""
    return secret



