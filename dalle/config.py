# Copyright (c) 2022 Forest Contact
import os
import functools
import logging
from typing import Optional, cast


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
