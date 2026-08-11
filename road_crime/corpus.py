"""Fetch clips from public repositories and turn them into track dumps.

The point is to remove the human from the loop. Nobody should be
downloading a video to a laptop and re-uploading it to Drive to get one
clip into the corpus; a source is named once and the clips arrive.

What actually gets kept is the dump, not the video. A clip is roughly
20 MB and needs a GPU to process; its dump is roughly 200 KB, needs
nothing, and is everything a violation module ever sees (CLAUDE.md
principle 5). So video is fetched once on a machine that does not care --
the Colab runtime -- and only dumps come back and go into `fixtures/`,
where they can be replayed for free forever and shared through git.

    from corpus import fetch, build_dumps

    clips = fetch("url:https://media.roboflow.com/.../vehicles.mp4")
    clips = fetch("hf:dgural/bdd100k", limit=10)
    clips = fetch("kaggle:deeplyft/driving-video-subset-50-with-object-tracking",
                  limit=10)
    build_dumps(clips, out_dir="fixtures")

For measuring false positives none of these sources needs labels. Ordinary
driving footage contains no wrong-way driving to any useful approximation,
so every alert raised on it is one we should not have raised. Positives
come from the injection pass in `evaluate.py`, which replays a real
trajectory backwards.
"""

import os
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DEFAULT_DEST = "clips"


def _video_files(root: str, limit: Optional[int]) -> List[str]:
    """Every video under `root`, deepest-first order made stable by sorting."""
    found: List[str] = []
    for directory, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.lower().endswith(VIDEO_SUFFIXES):
                found.append(os.path.join(directory, filename))
    found.sort()
    return found[:limit] if limit else found


def fetch_url(url: str, dest: str = DEFAULT_DEST) -> List[str]:
    """Download one clip by URL. No dependencies, no credentials."""
    os.makedirs(dest, exist_ok=True)
    name = os.path.basename(url.split("?")[0]) or "clip.mp4"
    path = os.path.join(dest, name)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        print("cached:", path)
        return [path]
    print("downloading", url)
    request = urllib.request.Request(url, headers={"User-Agent": "corpus.py"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as handle:
        handle.write(response.read())
    print("  -> {0} ({1} KB)".format(path, os.path.getsize(path) // 1024))
    return [path]


def extract_videos(
    archive: str, dest: str, limit: Optional[int] = None
) -> List[str]:
    """Pull up to `limit` video files out of a .tar or .zip.

    Driving datasets are routinely shipped as WebDataset tarballs rather
    than loose files, so a fetcher that only understands loose files finds
    nothing in most of them.
    """
    os.makedirs(dest, exist_ok=True)
    paths: List[str] = []

    if archive.lower().endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive) as bundle:
            names = [
                name for name in sorted(bundle.namelist())
                if name.lower().endswith(VIDEO_SUFFIXES)
            ]
            for name in names[:limit] if limit else names:
                target = os.path.join(dest, os.path.basename(name))
                with bundle.open(name) as source, open(target, "wb") as handle:
                    handle.write(source.read())
                paths.append(target)
        return paths

    import tarfile

    with tarfile.open(archive) as bundle:
        for member in bundle:
            if limit and len(paths) >= limit:
                break
            if not member.isfile():
                continue
            if not member.name.lower().endswith(VIDEO_SUFFIXES):
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            target = os.path.join(dest, os.path.basename(member.name))
            with source, open(target, "wb") as handle:
                handle.write(source.read())
            paths.append(target)
    return paths


def fetch_huggingface(
    repo_id: str, dest: str = DEFAULT_DEST, limit: Optional[int] = None
) -> List[str]:
    """Pull clips from a Hugging Face dataset repository.

    Public repositories need no credentials, which makes this the
    lowest-friction source. Only the files actually needed are downloaded,
    never the whole repository -- driving datasets run to hundreds of
    gigabytes.

    Loose video files are preferred. Failing that, archives are searched,
    because most driving datasets ship WebDataset tarballs. Note that an
    archive is all-or-nothing: `limit` bounds how many clips are extracted,
    not how much is downloaded.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    names = sorted(list_repo_files(repo_id, repo_type="dataset"))
    videos = [name for name in names if name.lower().endswith(VIDEO_SUFFIXES)]

    os.makedirs(dest, exist_ok=True)
    if videos:
        chosen = videos[:limit] if limit else videos
        paths = []
        for name in chosen:
            print("fetching", name)
            paths.append(
                hf_hub_download(repo_id, name, repo_type="dataset", local_dir=dest)
            )
        return paths

    archives = [
        name for name in names
        if name.lower().endswith((".tar", ".tar.gz", ".tgz", ".zip"))
    ]
    if not archives:
        raise RuntimeError(
            "{0} contains neither video files nor archives. Many driving "
            "datasets ship still frames instead of clips -- check the "
            "repository's file list before assuming this source fits.".format(repo_id)
        )

    archive_name = archives[0]
    print("no loose videos; downloading archive {0} (whole file)".format(archive_name))
    archive = hf_hub_download(repo_id, archive_name, repo_type="dataset", local_dir=dest)
    paths = extract_videos(archive, dest, limit)
    if not paths:
        raise RuntimeError(
            "No video files inside {0} either.".format(archive_name)
        )
    return paths


def fetch_kaggle(
    dataset: str, dest: str = DEFAULT_DEST, limit: Optional[int] = None
) -> List[str]:
    """Pull video files from a Kaggle dataset.

    Needs a one-time API token (kaggle.json). That is the only manual step
    in this module, and it is paid once per machine rather than once per
    clip -- which is the whole point.
    """
    import kagglehub

    root = kagglehub.dataset_download(dataset)
    print("kaggle dataset at", root)
    return _video_files(root, limit)


def fetch(
    source: str, dest: str = DEFAULT_DEST, limit: Optional[int] = None
) -> List[str]:
    """Fetch clips from a prefixed source string.

    `url:https://...`, `hf:owner/name`, `kaggle:owner/name`, or a local
    path, which is simply scanned. Returns local paths on whatever machine
    this runs on -- normally the Colab runtime.
    """
    if source.startswith("url:"):
        return fetch_url(source[4:], dest)
    if source.startswith("hf:"):
        return fetch_huggingface(source[3:], dest, limit)
    if source.startswith("kaggle:"):
        return fetch_kaggle(source[7:], dest, limit)
    if os.path.isdir(source):
        return _video_files(source, limit)
    if os.path.isfile(source):
        return [source]
    raise ValueError(
        "Unrecognised source {0!r}. Use url:, hf:, kaggle:, or a path.".format(source)
    )


def build_dumps(
    clips: Sequence[str],
    out_dir: str = "fixtures",
    limit_frames: Optional[int] = None,
    **run_kwargs: Any
) -> Dict[str, Optional[str]]:
    """Run the perception pipeline over each clip and keep only the dumps.

    Imports `pipeline` lazily: this module is useful on a laptop for
    inspecting what a source contains, and that should not require torch.

    A clip that fails is reported and skipped rather than killing the
    batch. Corpus building runs unattended over many clips, and one
    unreadable file should not cost the other forty.
    """
    from road_crime.pipeline import run

    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Optional[str]] = {}
    for index, clip in enumerate(clips, start=1):
        stem = os.path.splitext(os.path.basename(clip))[0]
        dump = os.path.join(out_dir, "{0}.jsonl".format(stem))
        if os.path.isfile(dump):
            print("[{0}/{1}] {2}: dump exists, skipping".format(index, len(clips), stem))
            results[clip] = dump
            continue
        print("[{0}/{1}] {2}".format(index, len(clips), stem))
        try:
            run(video=clip, dump_tracks=dump, limit_frames=limit_frames, **run_kwargs)
            results[clip] = dump
        except Exception as error:  # noqa: BLE001 - one bad clip must not stop the batch
            print("   FAILED: {0}: {1}".format(type(error).__name__, error))
            results[clip] = None
    made = [value for value in results.values() if value]
    print("\n{0} dump(s) in {1}/".format(len(made), out_dir))
    return results
