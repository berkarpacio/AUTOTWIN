"""
prop_database.py - Parse UIUC Propeller Database metadata from file names.

This module provides functionality to scan the UIUC Propeller Database
directory structure and parse propeller metadata from file names.

The UIUC Propeller Data Site is a tremendous resource maintained by
Prof. Michael Selig and students at UIUC. A large number of propellers
suitable for application to radio-control aircraft and small UAV's have
been tested in the wind tunnel with the results reported in the data site.

https://m-selig.ae.illinois.edu/props/propDB.html

The complete site may be downloaded in one large file here:
https://m-selig.ae.illinois.edu/props/download/UIUC-propDB.zip

This program expects you to have downloaded this file and un-zipped it
to some suitable location. This script expects to run from that location.

Based on propDataBase.m by Rob McDonald (rob.a.mcdonald@gmail.com)
Original MATLAB version: 17 February 2021 v. 1.0

Python conversion: 2025
"""

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Optional


@dataclass
class PropellerEntry:
    """
    Represents a propeller entry in the UIUC database.

    Numeric meta-data fields use NaN for not applicable or not specified.
    String/list fields use empty string/list for not applicable.

    Attributes:
        ident: Identifier string used to group data files
        diam: Diameter (inches)
        pitch: Pitch (inches)
        deg: Setting for ground-adjustable blades
        nblade: Number of blades for variable-blade props
        tracpush: Flag 1=Tractor, 2=Pusher
        specimen: Specimen number for repeated cases
        vnum: Volume number
        model: Size and model string as parsed
        vname: Volume name
        mfg: Manufacturer
        fname: List of all data file names
        typflag: File type flags (0=Perf, 1=Static, 2=Geom, 3=Thick)
        rpmv: RPM vector for wind-on data
        perf: List of wind-on performance file names
        static: List of static performance file names
        geom: List of geometry file names
        thick: List of thickness file names
        front: List of front image file names
        side: List of side image file names
    """
    ident: str = ""
    diam: float = float('nan')
    pitch: float = float('nan')
    deg: float = float('nan')
    nblade: float = float('nan')
    tracpush: float = float('nan')
    specimen: float = float('nan')
    vnum: int = 0
    model: str = ""
    vname: str = ""
    mfg: str = ""
    fname: list = field(default_factory=list)
    typflag: list = field(default_factory=list)
    rpmv: list = field(default_factory=list)
    perf: list = field(default_factory=list)
    static: list = field(default_factory=list)
    geom: list = field(default_factory=list)
    thick: list = field(default_factory=list)
    front: list = field(default_factory=list)
    side: list = field(default_factory=list)


def _safe_float(s: str) -> Optional[float]:
    """Attempt to convert string to float, return None if not possible."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_int(s: str) -> Optional[int]:
    """Attempt to convert string to int, return None if not possible."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


@dataclass
class _FileEntry:
    """Internal class for tracking parsed file data."""
    fname: str = ""
    vname: str = ""
    vnum: int = 0
    mfg: str = ""
    sz: str = ""
    model: str = ""
    diam: float = float('nan')
    pitch: float = float('nan')
    deg: float = float('nan')
    nblade: float = float('nan')
    tracpush: float = float('nan')
    specimen: float = float('nan')
    typflag: float = float('nan')
    rpm: float = float('nan')
    testid: str = ""


@dataclass
class _ImageEntry:
    """Internal class for tracking parsed image file data."""
    fname: str = ""
    vname: str = ""
    vnum: int = 0
    mfg: str = ""
    sz: str = ""
    model: str = ""
    diam: float = float('nan')
    pitch: float = float('nan')
    deg: float = float('nan')
    nblade: float = float('nan')
    tracpush: float = float('nan')
    specimen: float = float('nan')
    ityp: float = float('nan')  # 0=front, 1=side


def prop_database(base_path: Optional[Path] = None) -> list[PropellerEntry]:
    """
    Parse UIUC Propeller Database metadata from file names.

    Scans all data file names in the database and parses the names into
    data fields for each propeller entry.

    Args:
        base_path: Path to the UIUC-propDB root directory. If None, uses
                   current working directory.

    Returns:
        List of PropellerEntry objects, one for each unique propeller.

    Example:
        >>> prop_db = prop_database()
        >>> print(f"Found {len(prop_db)} propellers")
        >>> prop = prop_db[0]
        >>> print(f"First prop: {prop.mfg} {prop.diam}x{prop.pitch}")
    """
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)

    # Generate list of volumes
    vols = sorted(base_path.glob('volume-*'))

    if not vols:
        return []

    # Collect file names from each volume
    fnames_by_vol: dict[str, list[str]] = {}
    ifnames_by_vol: dict[str, list[str]] = {}

    for vol in vols:
        vname = vol.name
        data_path = vol / 'data'

        # List all .txt files in volume
        if data_path.exists():
            fnames_by_vol[vname] = [f.stem for f in sorted(data_path.glob('*.txt'))]
        else:
            fnames_by_vol[vname] = []

        # List all image files
        photo_path = vol / 'prop_photos'
        if photo_path.exists():
            jpg_files = [f.name for f in sorted(photo_path.glob('*.jpg'))]
            png_files = [f.name for f in sorted(photo_path.glob('*.png'))]
            ifnames_by_vol[vname] = jpg_files + png_files
        else:
            ifnames_by_vol[vname] = []

    # Parse data file names
    file_entries: list[_FileEntry] = []

    for vname, fnames in fnames_by_vol.items():
        for fname in fnames:
            entry = _parse_data_filename(fname, vname)
            file_entries.append(entry)

    # Parse image file names
    image_entries: list[_ImageEntry] = []

    for vname, ifnames in ifnames_by_vol.items():
        for ifname in ifnames:
            entry = _parse_image_filename(ifname, vname)
            image_entries.append(entry)

    # Generate identifiers for data files
    idents = [_make_ident(e) for e in file_entries]

    # Generate identifiers for image files
    iidents = [_make_ident_image(e) for e in image_entries]

    # Get unique identifiers while preserving order
    unique_idents = list(dict.fromkeys(idents))

    # Build PropellerEntry for each unique identifier
    prop_db: list[PropellerEntry] = []

    for uid in unique_idents:
        # Find all file entries with this identifier
        indices = [i for i, ident in enumerate(idents) if ident == uid]

        if not indices:
            continue

        # Use first entry for base metadata
        first_entry = file_entries[indices[0]]

        prop = PropellerEntry(
            ident=uid,
            diam=first_entry.diam,
            pitch=first_entry.pitch,
            deg=first_entry.deg,
            nblade=first_entry.nblade,
            tracpush=first_entry.tracpush,
            specimen=first_entry.specimen,
            vnum=first_entry.vnum,
            model=first_entry.model,
            vname=first_entry.vname,
            mfg=first_entry.mfg,
        )

        # Collect all files and categorize by type
        files = [file_entries[i].fname for i in indices]
        typs = [file_entries[i].typflag for i in indices]
        rpms = [file_entries[i].rpm for i in indices]

        prop.fname = files
        prop.typflag = [int(t) if not math.isnan(t) else t for t in typs]

        # Categorize files by type
        perf_files = []
        perf_rpms = []
        static_files = []
        geom_files = []
        thick_files = []

        for i, idx in enumerate(indices):
            typ = typs[i]
            if not math.isnan(typ):
                typ = int(typ)
                if typ == 0:  # Performance
                    perf_files.append(files[i])
                    rpm = rpms[i]
                    if not math.isnan(rpm):
                        perf_rpms.append(rpm)
                elif typ == 1:  # Static
                    static_files.append(files[i])
                elif typ == 2:  # Geometry
                    geom_files.append(files[i])
                elif typ == 3:  # Thickness
                    thick_files.append(files[i])

        prop.perf = perf_files
        prop.rpmv = perf_rpms
        prop.static = static_files
        prop.geom = geom_files
        prop.thick = thick_files

        # Find matching image files
        iindices = [i for i, iident in enumerate(iidents) if iident == uid]

        front_files = []
        side_files = []

        for idx in iindices:
            img_entry = image_entries[idx]
            if not math.isnan(img_entry.ityp):
                if int(img_entry.ityp) == 0:  # Front
                    front_files.append(img_entry.fname)
                else:  # Side
                    side_files.append(img_entry.fname)

        prop.front = front_files
        prop.side = side_files

        prop_db.append(prop)

    return prop_db


def _parse_data_filename(fname: str, vname: str) -> _FileEntry:
    """Parse a data file name into a _FileEntry."""
    entry = _FileEntry()
    entry.fname = fname
    entry.vname = vname

    # Parse volume number from vname
    vdat = vname.split('-')
    if len(vdat) > 1:
        vn = _safe_int(vdat[1])
        if vn is not None:
            entry.vnum = vn

    strs = fname.split('_')
    entry.mfg = strs[0] if strs else ""

    if len(strs) > 2:
        entry.sz = strs[1]

        dp = entry.sz.split('x')
        d = _safe_float(dp[0])
        p = float('nan')

        if d is None:
            entry.model = dp[0]
            d = float('nan')

            # Check for tractor/pusher with specimen notation
            if len(dp[0]) < 5 and 'p' in dp[0]:
                ds = dp[0].split('p')
                d_val = _safe_float(ds[0])
                if d_val is not None:
                    d = d_val
                if len(ds) > 1:
                    spec = _safe_float(ds[1])
                    if spec is not None:
                        entry.specimen = spec
                entry.tracpush = 2  # Pusher
            elif len(dp[0]) < 5 and 't' in dp[0]:
                ds = dp[0].split('t')
                d_val = _safe_float(ds[0])
                if d_val is not None:
                    d = d_val
                if len(ds) > 1:
                    spec = _safe_float(ds[1])
                    if spec is not None:
                        entry.specimen = spec
                entry.tracpush = 1  # Tractor
        else:
            if len(dp) > 1:
                p_val = _safe_float(dp[1])
                if p_val is not None:
                    p = p_val

        # Handle special cases for dimensions
        if entry.mfg == 'ancf':
            # Half-inch increments were specified without a decimal
            # For example, d=12.5 listed as 125
            if not math.isnan(d) and d > 60:
                d = d / 10
            if not math.isnan(p) and p > 30:
                p = p / 10
        elif not math.isnan(d) and d > 30:
            # Detect cases with dimensions in mm, convert to inch
            d = d / 25.4
            if not math.isnan(p):
                p = p / 25.4

        entry.diam = d
        entry.pitch = p

        # Attempt to detect entry type
        typ = strs[2] if len(strs) > 2 else ""

        # Tracker for additional fields
        iadd = 0

        if 'deg' in typ:
            deg_val = _safe_float(typ.replace('deg', ''))
            if deg_val is not None:
                entry.deg = deg_val
            iadd += 1
            typ = strs[3 + iadd - 1] if len(strs) > 3 + iadd - 1 else ""

        # Blade count (needs to come after deg)
        if len(typ) == 2 and 'b' in typ:
            nb = _safe_float(typ.replace('b', ''))
            if nb is not None:
                entry.nblade = nb
            iadd += 1
            typ = strs[3 + iadd - 1] if len(strs) > 3 + iadd - 1 else ""

        if 'spec' in typ:
            spec_val = _safe_float(typ.replace('spec', ''))
            if spec_val is not None:
                entry.specimen = spec_val
            iadd += 1
            typ = strs[3 + iadd - 1] if len(strs) > 3 + iadd - 1 else ""

        # Check type
        if typ == 'geom':
            entry.typflag = 2
        elif typ == 'static':
            entry.typflag = 1
            if len(strs) > 4 + iadd - 1:
                entry.testid = strs[4 + iadd - 1]
        else:  # Normal performance case
            entry.typflag = 0
            if len(strs) > 3 + iadd - 1:
                entry.testid = strs[3 + iadd - 1]
            if len(strs) > 4 + iadd - 1:
                rpm_val = _safe_float(strs[4 + iadd - 1])
                if rpm_val is not None:
                    entry.rpm = rpm_val

    else:  # Handle case where geom and thick are specified without type
        if len(strs) > 1:
            typ = strs[1]
            if typ == 'geom':
                entry.typflag = 2
            elif typ == 'thick':
                entry.typflag = 3

    return entry


def _parse_image_filename(ifname: str, vname: str) -> _ImageEntry:
    """Parse an image file name into an _ImageEntry."""
    entry = _ImageEntry()
    entry.fname = ifname
    entry.vname = vname

    # Parse volume number
    vdat = vname.split('-')
    if len(vdat) > 1:
        vn = _safe_int(vdat[1])
        if vn is not None:
            entry.vnum = vn

    # Image filenames use '-' as delimiter
    strs = ifname.split('-')

    if len(strs) >= 2:
        prefix = strs[0]
        viewdir = strs[1]

        if 'front' in viewdir:
            entry.ityp = 0
        else:
            entry.ityp = 1

        # Parse prefix for manufacturer and size
        prefix_parts = prefix.split('_')
        if prefix_parts:
            entry.mfg = prefix_parts[0]

        if len(prefix_parts) > 1:
            entry.sz = prefix_parts[1]

            dp = entry.sz.split('x')
            d = _safe_float(dp[0])
            p = float('nan')

            if d is None:
                entry.model = dp[0]
                d = float('nan')

                if len(dp[0]) < 5 and 'p' in dp[0]:
                    ds = dp[0].split('p')
                    d_val = _safe_float(ds[0])
                    if d_val is not None:
                        d = d_val
                    if len(ds) > 1:
                        spec = _safe_float(ds[1])
                        if spec is not None:
                            entry.specimen = spec
                    entry.tracpush = 2
                elif len(dp[0]) < 5 and 't' in dp[0]:
                    ds = dp[0].split('t')
                    d_val = _safe_float(ds[0])
                    if d_val is not None:
                        d = d_val
                    if len(ds) > 1:
                        spec = _safe_float(ds[1])
                        if spec is not None:
                            entry.specimen = spec
                    entry.tracpush = 1
            else:
                if len(dp) > 1:
                    p_val = _safe_float(dp[1])
                    if p_val is not None:
                        p = p_val

            # Handle special dimension cases
            if entry.mfg == 'ancf':
                if not math.isnan(d) and d > 60:
                    d = d / 10
                if not math.isnan(p) and p > 30:
                    p = p / 10
            elif not math.isnan(d) and d > 30:
                d = d / 25.4
                if not math.isnan(p):
                    p = p / 25.4

            entry.diam = d
            entry.pitch = p

            # Check for additional fields
            if len(prefix_parts) > 2:
                addfield = prefix_parts[2]

                if 'deg' in addfield:
                    deg_val = _safe_float(addfield.replace('deg', ''))
                    if deg_val is not None:
                        entry.deg = deg_val

                if len(addfield) == 2 and 'b' in addfield:
                    nb = _safe_float(addfield.replace('b', ''))
                    if nb is not None:
                        entry.nblade = nb

                if 'spec' in addfield:
                    spec_val = _safe_float(addfield.replace('spec', ''))
                    if spec_val is not None:
                        entry.specimen = spec_val

    return entry


def _make_ident(entry: _FileEntry) -> str:
    """Create identifier string from file entry."""
    parts = [entry.mfg]

    if not math.isnan(entry.diam):
        parts.append(f"_{entry.diam}")

    if not math.isnan(entry.pitch):
        parts.append(f"x{entry.pitch}")

    if not math.isnan(entry.deg):
        parts.append(f"_{entry.deg}deg")

    if not math.isnan(entry.nblade):
        parts.append(f"_{int(entry.nblade)}b")

    if not math.isnan(entry.tracpush):
        if entry.tracpush == 1:
            parts.append("_t")
        else:
            parts.append("_p")

    if not math.isnan(entry.specimen):
        parts.append(f"_spec{int(entry.specimen)}")

    return ''.join(parts)


def _make_ident_image(entry: _ImageEntry) -> str:
    """Create identifier string from image entry."""
    parts = [entry.mfg]

    if not math.isnan(entry.diam):
        parts.append(f"_{entry.diam}")

    if not math.isnan(entry.pitch):
        parts.append(f"x{entry.pitch}")

    if not math.isnan(entry.deg):
        parts.append(f"_{entry.deg}deg")

    if not math.isnan(entry.nblade):
        parts.append(f"_{int(entry.nblade)}b")

    if not math.isnan(entry.tracpush):
        if entry.tracpush == 1:
            parts.append("_t")
        else:
            parts.append("_p")

    if not math.isnan(entry.specimen):
        parts.append(f"_spec{int(entry.specimen)}")

    return ''.join(parts)


if __name__ == '__main__':
    # Example usage
    print("Parsing propeller database...")
    prop_db = prop_database()

    if prop_db:
        print(f"Found {len(prop_db)} propellers")

        # Show first few entries
        for i, prop in enumerate(prop_db[:5]):
            print(f"\n[{i}] {prop.ident}")
            print(f"    Manufacturer: {prop.mfg}")
            print(f"    Diameter: {prop.diam} in")
            print(f"    Pitch: {prop.pitch} in")
            print(f"    Volume: {prop.vname}")
            print(f"    Performance files: {len(prop.perf)}")
            print(f"    Static files: {len(prop.static)}")
            print(f"    Geometry files: {len(prop.geom)}")
    else:
        print("No propellers found. Make sure to run from the UIUC-propDB directory.")