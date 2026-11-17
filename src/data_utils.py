"""
File:    src/data_utils.py
Purpose: Reusable helpers for Part 1 (data exploration and data quality).

SAFETY
    Every function here only READS a file's bytes, using ``Path.read_bytes()``
    (the same as ``open(path, "rb").read()``) or ``pefile``, which parses
    headers without running anything. No function executes, opens, or copies a
    sample. The inventory table produced here contains only numbers, flags and
    hashes, never file contents, so it is safe to save or download.

HOW IT IS USED (in the Part 1 notebook)
    from src.data_utils import build_inventory, add_format_column, add_pe_details, CleaningFunnel

    df = build_inventory("/content/DikeDataset/files")   # one row per file
    df = add_format_column(df)                            # PE / OLE / MZ only / other
    df = add_pe_details(df)                               # header facts for PE files
    funnel = CleaningFunnel()                             # tracks counts per cleaning step

    Later, build_dataset.py can reuse inspect_file() to verify the SHA-256 of
    each downloaded file against your manifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants: "magic bytes" and header values used to recognise file types
# ---------------------------------------------------------------------------

#: Every DOS/Windows executable starts with these 2 bytes ("MZ").
MZ_MAGIC = b"MZ"

#: A real Windows PE file has these 4 bytes ("PE" + two zero bytes) at the
#: position stored in bytes 60-63 of the file (the "e_lfanew" field).
PE_SIGNATURE = b"PE\x00\x00"

#: Byte position of the e_lfanew field, and its size in bytes.
E_LFANEW_OFFSET = 60
E_LFANEW_SIZE = 4

#: Microsoft Office (OLE / Compound File) documents start with these 8 bytes.
OLE_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")

#: How many leading bytes to hash for the rough near-duplicate check.
HEAD_BYTES = 4096

#: FILE_HEADER.Machine value meaning "64-bit x86 (AMD64)".
IMAGE_FILE_MACHINE_AMD64 = 0x8664

#: Index of the ".NET (CLR) runtime header" entry in the PE data directory.
#: If this entry is non-empty, the file is a .NET assembly.
IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR = 14

#: Folder name -> label. 0 = benign, 1 = malware (the assignment's convention).
DEFAULT_FOLDERS = (("benign", 0), ("malware", 1))

# ---------------------------------------------------------------------------
# 1. Inventory: one row of facts per file
# ---------------------------------------------------------------------------


def inspect_file(path: str | Path, label: int) -> dict:
    """Read one file's bytes (never execute it) and return basic facts about it.

    Parameters
    ----------
    path : str or Path
        Location of the sample on the Linux machine (e.g. under /content/).
    label : int
        The label given by the folder the file came from: 0 = benign, 1 = malware.

    Returns
    -------
    dict with these keys (each becomes a column of the inventory table):
        path         full path to the file (as text)
        name         file name only (DikeDataset names files by their hash)
        label        0 = benign, 1 = malware (from the folder)
        size_bytes   file size in bytes
        sha256       SHA-256 of the whole file -> used to find exact duplicates
        md5          MD5 of the whole file -> lets you compare against file names
                     if the dataset names files by MD5 instead of SHA-256
        head_sha256  SHA-256 of only the first 4 KB -> rough near-duplicate check
        magic        first 8 bytes as hex text -> identifies the file format
        is_mz        True if the file starts with "MZ" (DOS/Windows executable)
        has_pe_sig   True if "PE\\0\\0" sits where bytes 60-63 say it should,
                     i.e. it is a real Windows PE file, not just an old DOS program
    """
    path = Path(path)
    data = path.read_bytes()  # read the raw bytes only; nothing is executed

    info = {
        "path": str(path),
        "name": path.name,
        "label": label,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "md5": hashlib.md5(data).hexdigest(),
        "head_sha256": hashlib.sha256(data[:HEAD_BYTES]).hexdigest(),
        "magic": data[:8].hex(),
        "is_mz": data[:2] == MZ_MAGIC,
        "has_pe_sig": False,
    }

    # A file needs at least 64 bytes to contain the e_lfanew field.
    if info["is_mz"] and len(data) >= E_LFANEW_OFFSET + E_LFANEW_SIZE:
        # Bytes 60-63 hold the position of the PE header (little-endian integer).
        pe_offset = int.from_bytes(
            data[E_LFANEW_OFFSET:E_LFANEW_OFFSET + E_LFANEW_SIZE], "little"
        )
        # Slicing past the end of the file just returns fewer bytes, so a bogus
        # offset safely results in False instead of an error.
        info["has_pe_sig"] = data[pe_offset:pe_offset + 4] == PE_SIGNATURE

    return info


def build_inventory(root: str | Path, folders=DEFAULT_FOLDERS) -> pd.DataFrame:
    """Run inspect_file() on every file in the dataset and return one table.

    Parameters
    ----------
    root : str or Path
        Folder that contains the class sub-folders,
        e.g. "/content/DikeDataset/files".
    folders : sequence of (folder_name, label) pairs
        Which sub-folders to read and which label each one gets.
        Default: ("benign", 0) and ("malware", 1).

    Returns
    -------
    pandas.DataFrame
        One row per file, with the columns described in inspect_file().
        Files are visited in sorted order so the table is reproducible.

    Notes
    -----
    Reads every file once, so it takes a few minutes for ~12,000 files.
    Run it once per Colab session and reuse the DataFrame afterwards.
    """
    root = Path(root)
    paths = [
        (p, label)
        for folder, label in folders
        for p in sorted((root / folder).iterdir())
        if p.is_file()
    ]

    try:  # show a progress bar if tqdm is available (it is in Colab)
        from tqdm.auto import tqdm
        paths = tqdm(paths, desc="Inspecting files")
    except ImportError:
        pass

    return pd.DataFrame([inspect_file(p, label) for p, label in paths])


# ---------------------------------------------------------------------------
# 2. File format: PE vs OLE vs everything else
# ---------------------------------------------------------------------------

def file_format(row) -> str:
    """Classify one inventory row by its magic bytes (not by its extension).

    Returns one of:
        "OLE"            Microsoft Office document (starts with D0 CF 11 E0 ...)
        "PE"             real Windows executable (MZ + PE signature)
        "MZ only (DOS)"  starts with MZ but has no PE header: old DOS program
                         or a damaged file
        "other"          anything else (unknown or corrupted)
    """
    if row["magic"].startswith(OLE_MAGIC.hex()):
        return "OLE"
    if row["has_pe_sig"]:
        return "PE"
    if row["is_mz"]:
        return "MZ only (DOS)"
    return "other"


def add_format_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the inventory with a new "format" column (see file_format)."""
    df = df.copy()
    df["format"] = df.apply(file_format, axis=1)
    return df


# ---------------------------------------------------------------------------
# 3. PE header details: corruption checks and "where do files come from?"
# ---------------------------------------------------------------------------

def pe_details(path: str | Path, size_bytes: int) -> dict:
    """Parse a PE file's headers with pefile and return quality/origin facts.

    pefile only reads and interprets the header structures; it never runs the
    program. fast_load=True skips the slower, deeper parsing we don't need.

    Parameters
    ----------
    path : str or Path
        Location of a file whose format is "PE".
    size_bytes : int
        The file's real size (from the inventory), used for the truncation check.

    Returns
    -------
    dict with these keys:
        parse_ok      False if pefile could not parse the headers at all
        parse_error   short error text when parse_ok is False, otherwise None
        has_warnings  True if pefile noticed something odd while parsing
        truncated     True if a section claims to extend past the end of the
                      file, i.e. the file is cut short or deliberately malformed
        is_64bit      True for 64-bit (AMD64) executables, False for others
        is_dll        True for libraries (.dll), False for programs (.exe)
        is_dotnet     True for .NET assemblies, whose code is bytecode rather
                      than machine code

        is_64bit, is_dll and is_dotnet are only filled in when parse_ok is True.

    Why it matters
    --------------
    - Parse failures and truncation are your "corrupted file" evidence. Compare
      them per class: malware is sometimes malformed on purpose, so dropping all
      failures could remove mostly malware.
    - is_64bit / is_dll / is_dotnet show whether benign and malware files differ
      in *kind*. If they do, a model can learn that difference as a shortcut
      instead of learning what makes a file malicious.
    """
    import pefile  # imported here so this module still loads where pefile isn't installed

    result = {
        "parse_ok": False, "parse_error": None, "has_warnings": None,
        "truncated": None, "is_64bit": None, "is_dll": None, "is_dotnet": None,
    }
    try:
        pe = pefile.PE(str(path), fast_load=True)
    except pefile.PEFormatError as e:
        result["parse_error"] = f"PEFormatError: {str(e)[:80]}"
        return result
    except Exception as e:  # malformed files can trip other errors in the parser
        result["parse_error"] = f"{type(e).__name__}: {str(e)[:80]}"
        return result

    try:
        data_dirs = pe.OPTIONAL_HEADER.DATA_DIRECTORY
        result.update(
            parse_ok=True,
            has_warnings=bool(pe.get_warnings()),
            # PointerToRawData = where a section starts in the file,
            # SizeOfRawData = how many bytes it says it has.
            truncated=any(
                s.PointerToRawData + s.SizeOfRawData > size_bytes for s in pe.sections
            ),
            is_64bit=pe.FILE_HEADER.Machine == IMAGE_FILE_MACHINE_AMD64,
            is_dll=pe.is_dll(),
            # Some malformed files have fewer data directories than normal,
            # so check the length before reading entry 14.
            is_dotnet=(
                len(data_dirs) > IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR
                and data_dirs[IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR].VirtualAddress != 0
            ),
        )
    except Exception as e:
        result["parse_ok"] = False
        result["parse_error"] = f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        pe.close()
    return result


def add_pe_details(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the inventory with pe_details() columns added.

    Only rows whose "format" is "PE" are parsed (run add_format_column first).
    Other rows get empty values (NaN/None) in the new columns.
    """
    df = df.copy()
    pe_rows = df[df["format"] == "PE"]
    details = pd.DataFrame(
        [pe_details(p, s)
         for p, s in zip(pe_rows["path"], pe_rows["size_bytes"])],
        index=pe_rows.index,
    )
    return df.join(details)


# ---------------------------------------------------------------------------
# 4. Cleaning funnel: how many files of each class survive each step
# ---------------------------------------------------------------------------

class CleaningFunnel:
    """Record how many benign and malware files remain after each cleaning step.

    Usage
    -----
        funnel = CleaningFunnel()
        funnel.log("Start (all files)", df)
        clean = df[df["format"] == "PE"]
        funnel.log("Drop OLE / non-PE", clean)
        ...
        funnel.table()   # -> DataFrame you can show in the notebook and video

    The table shows, per step, how many files are left in each class, how many
    were removed, and the malware:benign ratio, so you can see whether a
    cleaning decision changed the class balance.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def log(self, step: str, df: pd.DataFrame) -> None:
        """Record the class counts of `df` after the step named `step`."""
        benign = int((df["label"] == 0).sum())
        malware = int((df["label"] == 1).sum())
        self._rows.append(
            {"step": step, "benign_left": benign, "malware_left": malware})

    def table(self) -> pd.DataFrame:
        """Return the funnel as a DataFrame with removed counts and class ratio."""
        t = pd.DataFrame(self._rows)
        if t.empty:
            return t
        t["benign_removed"] = (-t["benign_left"].diff()).fillna(0).astype(int)
        t["malware_removed"] = (-t["malware_left"].diff()
                                ).fillna(0).astype(int)
        t["malware_per_benign"] = (
            t["malware_left"] / t["benign_left"]).round(2)
        return t[["step", "benign_removed", "malware_removed",
                  "benign_left", "malware_left", "malware_per_benign"]]
