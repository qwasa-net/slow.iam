import argparse
import logging
import os
import re
import sys
import urllib.parse

import torch

from ihelp.config import Configuration
from ihelp.helpers import setup_logging, str_eclipse
from ihelp.media import load_data

from .detect import Detector

log = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args(with_command=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        type=str,
        choices=["index", "query", "cut"],
        nargs="?" if with_command else None,
        default=with_command,
    )
    parser.add_argument(
        "src",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--src-filter-regex",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--src-filter",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save-img",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/ireqs/",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--limit-per-item",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
    )
    parser.add_argument(
        "--database",
        "-db",
        type=str,
        default="",
    )
    parser.add_argument(
        "--database-prefix",
        type=str,
        default="",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="face",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
    )
    args, _ = parser.parse_known_args()

    args.src = [urllib.parse.unquote(s[7:]) if s.startswith("file://") else s for s in args.src]
    args.src = [os.path.abspath(s) for s in args.src]

    if args.src_filter:
        if args.src_filter_regex:
            args.src_filter = re.compile(args.src_filter, flags=re.IGNORECASE)
        else:
            args.src_filter = re.compile(re.escape(args.src_filter), flags=re.IGNORECASE)

    if args.tag:
        args.tag = re.sub(r"[^a-z0-9_]+", "_", args.tag.strip().lower())

    setup_logging(args)

    config = Configuration()
    config.configure(params=vars(args), env_use=True, env_prefix="IREQS")
    log.info("config: %s", args)

    return config


def scan_and_detect(detector, config):
    for mi_c, mi in enumerate(load_data(config.src, config), start=1):
        log.debug("%s: %s", mi_c, mi)
        for det_c, det in enumerate(detector.detect(mi), start=1):
            yield det
            if det_c >= config.limit_per_item:
                log.info("per-item limit reached: %s detected at %s", det_c, mi)
                break
        if mi_c >= config.limit:
            log.info("items limit reached: %s", mi_c)
            break


def run(processors, config):
    detector = Detector(config)
    for det_c, det in enumerate(scan_and_detect(detector, config), start=1):
        name = f"{det.media_item.name}-{det_c}"
        log.info("№%s: name=%s %s", det_c, name, str_eclipse(det.media_item, 60))
        piper = None
        for processor in processors:
            piper = processor(name, det, piper)


def main():
    config = parse_args()
    if config.command == "index":
        from .index import main as main_index

        main_index(config)
    elif config.command == "query":
        from .query import main as main_query

        main_query(config)
    else:
        log.error("unknown command: %s", config.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
