import dataclasses
import enum
import logging
import os
import random
import re
import subprocess
import tempfile

from PIL import Image

from ihelp.helpers import str_eclipse

MPV_EXE_PATH = "/usr/bin/mpv"
FFMPEG_EXE_PATH = "/usr/bin/ffmpeg"

VIDEO_STEP_SEC = 59
VIDEO_CAPS_LIMIT = 123

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".webm", ".mov", ".flv", ".wmv", ".ts"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

TAG_REGEXP = re.compile(r"^[a-zA-Z0-9_]{1,64}", flags=re.IGNORECASE)

log = logging.getLogger()


class MediaItemType(enum.IntEnum):
    IMAGE = 1
    VIDEO = 2

    @classmethod
    def __missing__(cls, value):
        return cls.IMAGE

    def __repr__(self):
        return self.name.upper()

    def __str__(self):
        return self.name.upper()


@dataclasses.dataclass
class MediaItem:
    path: str
    opath: str | None = None
    particle: int | None = None
    type: MediaItemType = MediaItemType.IMAGE
    image: Image.Image | None = None
    tag: str | None = None

    @property
    def is_image(self):
        return self.type == MediaItemType.IMAGE

    @property
    def is_video(self):
        return self.type == MediaItemType.VIDEO

    @property
    def name(self):
        if self.particle is not None:
            return f"{self.tagname}-{self.particle}"
        return self.tagname

    @property
    def tagname(self):
        if self.tag:
            return self.tag
        # some_tag_name-990101-123456.jpg → some_tag_name
        if self.opath is None:
            return None
        bn = os.path.basename(self.opath)
        bnn, _ = os.path.splitext(bn)
        tag_mo = TAG_REGEXP.match(bnn)
        if tag_mo:
            return re.sub(r"[^a-z0-9_]+", "_", tag_mo.group(0).lower())
        return bnn

    def __str__(self):
        return f"MediaItem(name={self.name}, type={self.type}, opath={str_eclipse(self.opath, 60, 1 / 4)})"

    def __repr__(self):
        return self.__str__()


def save_image(path, data, image_format="JPEG", quality=95):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    data.save(path, format=image_format, quality=quality)


def load_data(src, config):
    for mi in get_paths_limited(src, config.limit, config.src_filter):
        if mi.is_video:
            yield from load_video_data(mi)
        elif mi.is_image:
            yield from load_image_data(mi)


def get_paths_limited(src, limit=0, src_filter=None):
    if isinstance(src, str):
        src = [src]
    if not isinstance(src, list):
        raise ValueError("src must be a string or a list of strings")
    paths = []
    for s in src:
        paths += list(get_paths(s, src_filter))
    if limit > 0 and len(paths) > limit:
        random.shuffle(paths)
        paths = paths[:limit]
    paths.sort(key=lambda x: x.opath)
    return paths


def get_paths(src, src_filter=None):
    mi_type = MediaItemType.IMAGE

    if os.path.isdir(src):
        for root, _dirs, filenames in os.walk(src):
            for filename in filenames:
                path = os.path.join(root, filename)
                yield from get_paths(path, src_filter)

    if os.path.isfile(src):
        if src_filter and src_filter.search(src) is None:
            return
        _, fext = os.path.splitext(src)
        if fext.lower() not in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS:
            return
        if fext.lower() in VIDEO_EXTENSIONS:
            mi_type = MediaItemType.VIDEO
        mi = MediaItem(src, src, type=mi_type)
        yield mi


def load_image_data(mi):
    try:
        img = Image.open(mi.path)
        mi.image = img
        yield mi
    except Exception as e:
        log.error("image load error `%.100s`: %s", mi.path, e)


def load_video_data(mi):
    yield from load_video_data_from_player(mi, player_cmd_mpv)


def guess_video_data_step(mi, tiny=0.25 * 1024 * 1024, huge=500 * 1024 * 1024):
    try:
        file_size = os.path.getsize(mi.path)
    except Exception:
        file_size = huge + 1
    if file_size < tiny:  # skip small <250KB
        return None
    return max(3, VIDEO_STEP_SEC // 3) if file_size < huge else VIDEO_STEP_SEC


def load_video_data_from_player(mi, player_cmd, sstep=None, limit=VIDEO_CAPS_LIMIT):
    with tempfile.TemporaryDirectory(suffix="-caps-maker-slow.iam", ignore_cleanup_errors=True) as tmpdir:
        os.chmod(tmpdir, 0o755)

        sstep = guess_video_data_step(mi)
        if sstep is None:
            return

        cmd = player_cmd(mi.path, tmpdir, sstep=sstep, limit=limit)

        try:
            log.info("calling caps maker: %s", cmd)
            subprocess.run(cmd, check=True)
        except Exception as e:
            log.error("caps maker error `%.100s`: %.1000s", mi.path, e)
            return

        paths = []
        for root, _dirs, filenames in os.walk(tmpdir):
            for filename in filenames:
                cpath = os.path.join(root, filename)
                paths.append(cpath)
        paths.sort()

        prev_file_size = None
        for i, path in enumerate(paths):
            file_size = os.path.getsize(path)
            if file_size == prev_file_size:
                continue  # skip duplicates, detected by size (¿bug?)
            prev_file_size = file_size
            mii = MediaItem(
                path,
                opath=mi.path,
                particle=mi.particle + i if mi.particle is not None else i,
                type=MediaItemType.IMAGE,
            )
            yield from load_image_data(mii)


def player_cmd_ffmpeg(vpath, tmpdir, sstep=59, limit=100):
    tmpfile_out = os.path.join(tmpdir, "f%06d.jpg")
    return [
        FFMPEG_EXE_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        vpath,
        "-vf",
        f"fps=1/{sstep}",
        tmpfile_out,
    ]


def player_cmd_mpv(vpath, tmpdir, sstep=39, limit=100):
    return [
        MPV_EXE_PATH,
        "--really-quiet",
        "--untimed",
        "--no-correct-pts",
        "--hr-seek=no",
        "--hr-seek-framedrop=yes",
        "--no-audio",
        "--slang=",
        "--hwdec=auto-safe",
        "--vo=image",
        "--vo-image-format=jpg",
        "--vo-image-jpeg-quality=98",
        f"--vo-image-outdir={tmpdir}",
        f"--sstep={sstep}",
        f"--frames={limit}",
        vpath,
    ]
