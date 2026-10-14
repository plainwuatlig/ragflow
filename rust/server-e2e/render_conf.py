#!/usr/bin/env python3
"""Renders upstream's docker/service_conf.yaml.template the way the image's
entrypoint does: every ${NAME:-default} becomes the environment's NAME, or the
default when NAME is unset or empty.

    render_conf.py service_conf.yaml.template > service_conf.yaml
"""
import os
import re
import sys

pattern = re.compile(r"\$\{([^:}]+):-([^}]*)\}")
for line in open(sys.argv[1], encoding="utf-8"):
    sys.stdout.write(pattern.sub(lambda m: os.environ.get(m.group(1)) or m.group(2), line))
