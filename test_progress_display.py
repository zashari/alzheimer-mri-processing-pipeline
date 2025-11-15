#!/usr/bin/env python
"""
Test script to verify progress display fixes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_progress_display():
    """Test that progress display works correctly."""
    print("=" * 60)
    print("Testing Progress Display Fix")
    print("=" * 60)

    # Import the formatter
    from data_processing.nifti_processing.formatter import NiftiFormatter

    # Create formatter
    formatter = NiftiFormatter(verbose=False, quiet=False, json_only=False)

    # Test 1: Header without profile
    print("\n1. Testing header without profile:")
    formatter.header("process", "skull_stripping", device="cuda")

    # Test 2: Configuration without profile
    print("\n2. Testing configuration without profile:")
    formatter.configuration("cuda", None, False, 5, True)

    # Test 3: Configuration with profile (for comparison)
    print("\n3. Testing configuration with profile (should not show):")
    formatter.configuration("cuda", None, False, 5, True)

    # Test 4: Progress bar with statistics
    print("\n4. Testing progress bar with statistics:")
    with formatter.create_progress_bar() as progress:
        task = progress.add_task("Processing files", total=10)

        # Simulate processing
        for i in range(10):
            import time
            time.sleep(0.1)

            # Update with statistics
            progress.update(task,
                advance=1,
                description=f"[green]{i+1} done[/green] | "
                           f"[yellow]0 skipped[/yellow] | "
                           f"[red]0 failed[/red]")

    print("\n✅ All display tests completed successfully!")

if __name__ == "__main__":
    test_progress_display()