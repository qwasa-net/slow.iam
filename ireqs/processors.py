import logging
import os
import sys
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageOps, ImageTk

from ihelp.helpers import str_eclipse
from ihelp.media import save_image

from .db import DB
from .detect import Detected

log = logging.getLogger(__name__)


class Processor:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        pass


class DBInsertProcessor(Processor):
    def __init__(self, config, db: DB) -> None:
        self.config = config
        self.db = db

    def __call__(self, name: str, det: Detected, rz: dict, *args, **kwargs):
        rz = rz or {}
        tag = str(self.config.tag or det.media_item.tagname or "-").lower()
        data = {
            "name": name,
            "path": det.media_item.opath,
            "det_score": float(det.score),
        } | rz
        embedding = det.vector
        pk = self.db.insert_vector(tag, data, embedding)
        log.info("inserted to db: `%s` with tag=`%s` pk=%s", name, tag, pk)
        return data


class DBQueryProcessor(Processor):
    def __init__(self, db: DB) -> None:
        self.db = db
        self.limit = 36

    def __call__(self, name: str, det: Detected, *args, **kwargs):
        results = self.db.search_similar(det.vector, limit=self.limit)
        tags = [r[1] for r in results]
        log.info("results for %s [%s]: %s", det.media_item, len(results), tags)
        return results


class SaveProcessor(Processor):
    def __init__(self, config) -> None:
        self.config = config
        self.magic = f"{int(time.time()) % (2**16):x}"

    def __call__(self, name: str, det: Detected, *args, **kwargs) -> dict:
        image = det.image or det.media_item.image
        if image is None:
            return {}
        fname = f"{name}_{self.magic}.jpg"
        subdir = str(self.config.tag or det.media_item.tagname).lower()
        path = os.path.join(self.config.save_dir, subdir, fname)
        save_image(path, image)
        log.info("saved `%s` to `%s`", str_eclipse(det.media_item, 60, 0.5), str_eclipse(path, 60, 0.5))
        return {"saved_path": path}


class ShowProcessor(Processor):
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, name: str, det: Detected, results, *args, **kwargs):
        if isinstance(results, list):
            paths = [(r[1], r[2]["saved_path"], r[3]) for r in results if r[2] and "saved_path" in r[2]]
        else:
            paths = []
        self.show(det, paths)
        return results

    def show(self, det, paths):
        root = tk.Tk()
        root.title("ireqs")
        root.attributes("-zoomed", True)
        root.update_idletasks()
        sw = root.winfo_screenwidth() - 20
        sh = root.winfo_screenheight() - 60

        pady = min(20, sh // 30)
        paddy = pady // 2

        top_frame = tk.Frame(root)
        top_frame.pack(side="top", pady=paddy)

        topih = sh // 3
        for img in filter(None, (det.media_item.image, det.image)):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, list):
                img = Image.fromarray(np.array(img))
            img = ImageOps.exif_transpose(img)
            img = img.resize((int(img.width * topih / img.height), topih))
            img_tk = ImageTk.PhotoImage(img)
            label = tk.Label(top_frame, image=img_tk)
            label._image = img_tk
            label.pack(side="left", padx=paddy, pady=paddy)

        bottom_frame = tk.Frame(root)
        bottom_frame.pack(side="top", fill="x", expand=True)

        cols = 12
        visible_rows = max(1, (len(paths) + 1) // cols)
        ch = int(sh * (2 / 3) / (visible_rows) - pady)
        thh = int(ch * 0.8)
        cw = int(sw / cols - paddy)
        thw = int(cw)

        igrid = tk.Frame(bottom_frame)
        igrid.pack(side="top", fill="x", expand=True)
        for col in range(cols):
            igrid.grid_columnconfigure(col, weight=1)

        labels = []
        for i, (label, path, dist) in enumerate(paths):
            if label not in labels:
                labels.append(label)
            cell = tk.Frame(igrid, width=cw, height=ch)
            if path and os.path.isfile(path):
                img = Image.open(path)
                if dist > 0.69:
                    img = img.convert("L").convert("RGB")
                img = ImageOps.exif_transpose(img)
                iw, ih = img.size
                ir = min(1, iw / ih)
                img = img.resize((int(min(thh * ir, thw)), thh))
                img_tk = ImageTk.PhotoImage(img)
                img_widget = tk.Label(cell, image=img_tk)
                img_widget._image = img_tk
                img_widget.pack(side="top", expand=True)
            tag_widget = tk.Label(cell, text=f"{label:.20s} {dist:.2f}", fg="black", font=("Arial", 8))
            tag_widget.pack(side="top", fill="both")
            cell.grid(row=i // cols, column=i % cols, padx=paddy, pady=paddy)

        labels_widget = tk.Label(bottom_frame, text=", ".join(labels), font=("Arial", 10))
        labels_widget.pack(side="top", pady=pady, padx=paddy)

        def on_key(event):
            if event.keysym == "Escape":
                root.destroy()
                sys.exit(0)
            elif event.char and event.char.isascii():
                root.destroy()

        root.bind("<Key>", on_key)
        root.mainloop()
