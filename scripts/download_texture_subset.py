#!/usr/bin/env python3
"""
Download a subset of CC0 textures for BlenderProc.

This script provides a workaround for the huge CC0 texture download by:
1. Providing instructions to manually download specific texture packs
2. Or using a simplified download approach

Since BlenderProc downloads all 56GB+ at once, this script helps users
get started with just essential textures.
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import List


# Recommended textures by ID from ambientCG for each subset
# These are high-quality, commonly used textures
RECOMMENDED_TEXTURES = {
    "minimal": [
        # ~5GB - Essential basics
        "Wood051", "Wood026", "Wood049", "Wood075",
        "Metal006", "Metal034", "Metal032",
        "Fabric002", "Fabric030", "Fabric013",
        "Concrete018", "Concrete025", "Concrete033",
        "Ground037", "Ground042", "Ground054",
        "Tiles074", "Tiles083", "Tiles036",
        "Plastic003", "Plastic008",
        "Stone004", "Stone020", "Stone033",
    ],
    "medium": [
        # ~15GB - Good variety (includes all minimal + more)
        "Wood051", "Wood026", "Wood049", "Wood075", "Wood081", "Wood062",
        "Metal006", "Metal034", "Metal032", "Metal018", "Metal027",
        "Fabric002", "Fabric030", "Fabric013", "Fabric039", "Fabric044",
        "Concrete018", "Concrete025", "Concrete033", "Concrete028", "Concrete043",
        "Ground037", "Ground042", "Ground054", "Ground038", "Ground049",
        "Tiles074", "Tiles083", "Tiles036", "Tiles073", "Tiles076",
        "Plastic003", "Plastic008", "Plastic006",
        "Stone004", "Stone020", "Stone033", "Stone038", "Stone041",
        "Brick037", "Brick043", "Brick068", "Brick062",
        "Leather011", "Leather018", "Leather026",
        "Marble012", "Marble006", "Marble019",
    ],
}


def print_color(text: str, color: str = "green"):
    """Print colored text."""
    colors = {
        "red": "\033[0;31m",
        "green": "\033[0;32m",
        "yellow": "\033[1;33m",
        "blue": "\033[0;34m",
        "cyan": "\033[0;36m",
        "nc": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['nc']}")


def show_manual_instructions(subset: str, output_path: Path):
    """Show instructions for manual texture download."""
    textures = RECOMMENDED_TEXTURES.get(subset, [])
    
    print_color("\n" + "="*70, "blue")
    print_color("  MANUAL TEXTURE DOWNLOAD INSTRUCTIONS", "blue")
    print_color("="*70, "blue")
    print()
    
    print_color("BlenderProc's CC0 download is 56GB+ which is impractical.", "yellow")
    print_color("Here's how to get started with a smaller subset:", "cyan")
    print()
    
    print_color("OPTION 1: Skip textures (RECOMMENDED for testing)", "green")
    print_color("-" * 50, "cyan")
    print("  Run: ./install.sh --skip-textures")
    print("  Result: Datasets use solid colors (still trains well!)")
    print("  Size: 0 GB")
    print()
    
    print_color("OPTION 2: Manual download (for production use)", "green")
    print_color("-" * 50, "cyan")
    print(f"  1. Visit: https://ambientcg.com/")
    print(f"  2. Search for and download these {len(textures)} textures:")
    print()
    
    # Group textures by category for clearer output
    categories = {}
    for tex in textures:
        # Extract category from texture ID (e.g., Wood051 -> Wood)
        category = ''.join([c for c in tex if not c.isdigit()])
        if category not in categories:
            categories[category] = []
        categories[category].append(tex)
    
    for category, tex_list in sorted(categories.items()):
        print(f"     {category}: {', '.join(tex_list)}")
    
    print()
    print(f"  3. Extract ZIP files to: {output_path}")
    print(f"  4. Directory structure should be:")
    print(f"     {output_path}/")
    print(f"       Wood051/")
    print(f"       Wood026/")
    print(f"       Metal006/")
    print(f"       ...")
    print()
    
    print_color("OPTION 3: Use environment textures (simpler)", "green")
    print_color("-" * 50, "cyan")
    print("  BlenderProc can use HDRI environment maps instead:")
    print("  1. Download 5-10 HDRIs from https://polyhaven.com/hdris")
    print("  2. Much smaller (~50-100MB total)")
    print("  3. Provides lighting + reflections")
    print()
    
    print_color("="*70, "blue")
    print()


def create_texture_index(output_path: Path):
    """Create an index file of available textures."""
    print_color("Creating texture index...", "blue")
    
    textures = []
    for item in output_path.iterdir():
        if item.is_dir():
            textures.append(item.name)
    
    index_file = output_path / "texture_index.json"
    with open(index_file, 'w') as f:
        json.dump({
            "texture_count": len(textures),
            "textures": sorted(textures),
        }, f, indent=2)
    
    print_color(f"✓ Created index: {index_file}", "green")
    print_color(f"  Found {len(textures)} textures", "cyan")


def main():
    parser = argparse.ArgumentParser(
        description="Helper for downloading CC0 texture subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script helps work around the massive 56GB+ CC0 texture download.

For most users, we recommend:
  1. Start with --skip-textures (solid colors work fine!)
  2. If needed, manually download 5-10 textures from ambientcg.com
  3. Or use HDRI environment maps instead (much smaller)

Examples:
  # Show instructions for minimal subset
  python download_texture_subset.py --subset minimal --output ./textures

  # Create index of existing textures
  python download_texture_subset.py --index ./cc0_textures
        """
    )
    
    parser.add_argument(
        "--subset",
        choices=["minimal", "medium"],
        help="Show instructions for this subset"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for textures"
    )
    
    parser.add_argument(
        "--index",
        type=Path,
        help="Create index of existing textures in directory"
    )
    
    args = parser.parse_args()
    
    if args.index:
        if not args.index.exists():
            print_color(f"Error: Directory not found: {args.index}", "red")
            return 1
        create_texture_index(args.index)
        return 0
    
    if not args.subset or not args.output:
        parser.print_help()
        return 1
    
    # Show manual instructions
    show_manual_instructions(args.subset, args.output)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Create a README in the texture directory
    readme_path = args.output / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"CC0 Texture Subset: {args.subset}\n")
        f.write("=" * 50 + "\n\n")
        f.write("This directory should contain texture folders downloaded from:\n")
        f.write("https://ambientcg.com/\n\n")
        f.write("Recommended textures:\n")
        for tex in RECOMMENDED_TEXTURES.get(args.subset, []):
            f.write(f"  - {tex}\n")
        f.write("\nEach texture should be in its own directory with texture files.\n")
    
    print_color(f"\n✓ Created: {readme_path}", "green")
    print_color(f"✓ Texture directory ready: {args.output}", "green")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
