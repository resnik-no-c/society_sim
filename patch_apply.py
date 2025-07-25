#!/usr/bin/env python3
"""
Patch applicator that automatically fixes malformed hunk headers by recalculating context counts,
then invokes GNU `patch` to apply the unified diff with optional fuzz, dry-run, and backups.

Usage:
    ./patch_apply.py [options] <patch-file>

Options:
    -p N, --strip N     strip N leading components from file paths (patch -p) (default: 1)
    -F N, --fuzz N      allow up to N lines of context mismatch (patch --fuzz) (default: 3)
    --backup            keep backups of original files (patch --backup)
    --no-backup         do NOT keep backups of original files (default behavior)
    --dry-run           preview changes without modifying any files (patch --dry-run)
    --quiet             suppress output and only return exit status
    -h, --help          show this help message and exit
"""
import argparse
import re
import subprocess
import sys
import tempfile
import os

# Regex to match unified diff hunk headers
HUNK_RE = re.compile(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)$')

def fix_patch_counts(lines):
    """Recalculate and correct hunk header counts based on actual hunk body."""
    fixed = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = HUNK_RE.match(line)
        if m:
            old_start, _, new_start, _, desc = m.groups()
            # collect the hunk body until next hunk or new file diff
            hunk_body = []
            i += 1
            while i < n and not HUNK_RE.match(lines[i]) and not lines[i].startswith('diff --git '):
                hunk_body.append(lines[i])
                i += 1
            # recount
            old_n = new_n = 0
            for h in hunk_body:
                if h.startswith(' '):
                    old_n += 1
                    new_n += 1
                elif h.startswith('-') and not h.startswith('---'):
                    old_n += 1
                elif h.startswith('+') and not h.startswith('+++'):
                    new_n += 1
            # write corrected header
            fixed.append(f"@@ -{old_start},{old_n} +{new_start},{new_n} @@{desc}\n")
            fixed.extend(hunk_body)
        else:
            fixed.append(line)
            i += 1
    return fixed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Apply a unified diff by auto‑fixing hunk headers and invoking GNU patch"
    )
    parser.add_argument('patch_file', help='Unified diff file to apply')
    parser.add_argument('-p', '--strip', type=int, default=1,
                        help='strip N leading components from file paths (default: 1)')
    parser.add_argument('-F', '--fuzz', type=int, default=3,
                        help='allow up to N lines of context mismatch (default: 3)')
    parser.add_argument('--backup', action='store_true',
                        help='keep backups of original files (adds .orig suffix)')
    parser.add_argument('--no-backup', dest='backup', action='store_false',
                        help='do NOT keep backups of original files (default)')
    parser.add_argument('--dry-run', action='store_true',
                        help='preview changes without modifying files')
    parser.add_argument('--quiet', action='store_true',
                        help='suppress output')
    args = parser.parse_args()

    # Read original patch
    if not os.path.isfile(args.patch_file):
        print(f"❌ Patch file not found: {args.patch_file}", file=sys.stderr)
        sys.exit(1)
    with open(args.patch_file, 'r', encoding='utf-8', errors='replace') as f:
        orig_lines = f.readlines()

    # Fix hunk headers
    fixed_lines = fix_patch_counts(orig_lines)

    # Write to temporary patch file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.patch', mode='w', encoding='utf-8')
    tmp.writelines(fixed_lines)
    tmp_path = tmp.name
    tmp.close()

    # Build GNU patch command
    cmd = [
        'patch',
        f'-p{args.strip}',
        f'--fuzz={args.fuzz}',
        '-i', tmp_path,
    ]
    if args.backup:
        cmd.append('--backup')
    if args.dry_run:
        cmd.append('--dry-run')

    if not args.quiet:
        print('🔧 Invoking:', ' '.join(cmd))

    # Execute patch
    try:
        proc = subprocess.run(
            cmd,
            stdout=(subprocess.DEVNULL if args.quiet else None),
            stderr=(subprocess.DEVNULL if args.quiet else None)
        )
    except FileNotFoundError:
        print("❌ Error: 'patch' command not found. Install GNU patch.", file=sys.stderr)
        os.unlink(tmp_path)
        sys.exit(1)

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except:
        pass

    # Exit with the patch return code
    sys.exit(proc.returncode)
