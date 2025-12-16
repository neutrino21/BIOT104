#!/usr/bin/env python3
"""
Interactive NPZ file viewer with genomic context
Usage: python npz_viewer.py file.npz [metadata.json]

-----------------------------------------------------------------------
Copyright (c) 2025 Sean Kiewiet. All rights reserved.
-----------------------------------------------------------------------
"""

import sys
import numpy as np
import json
from pathlib import Path

def view_npz(filename, metadata_file=None):
    """Load and display NPZ file contents interactively"""

    print("="*60)
    print(f"NPZ FILE: {filename}")
    print("="*60)

    # Load the file
    data = np.load(filename)

    # Try to load metadata for genomic context
    metadata = None
    if metadata_file:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        # Try to find matching JSON
        json_path = Path(filename).with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            print(f"Found metadata: {json_path}")

    # Show genomic context if available
    if metadata:
        print("\nGENOMIC CONTEXT:")
        print("-"*60)
        if 'chromosome' in metadata:
            print(f"  Chromosome: {metadata['chromosome']}")
        if 'start' in metadata and 'end' in metadata:
            print(f"  Region: {metadata['start']:,} - {metadata['end']:,}")
            print(f"  Size: {metadata['end'] - metadata['start']:,} bp")
        if 'gene' in metadata:
            print(f"  Gene: {metadata['gene']}")
        if 'ontology' in metadata:
            print(f"  Cell type: {metadata['ontology']}")

    # Show all arrays
    print(f"\nArrays found: {list(data.files)}")
    print("\nDETAILS:")
    print("-"*60)

    for i, name in enumerate(data.files, 1):
        arr = data[name]
        print(f"\n[{i}] '{name}':")
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")
        print(f"    Size: {arr.size} elements")
        print(f"    Bytes: {arr.nbytes:,} ({arr.nbytes/1024/1024:.2f} MB)")

        if arr.size > 0:
            print(f"    Min: {arr.min():.6f}")
            print(f"    Max: {arr.max():.6f}")
            print(f"    Mean: {arr.mean():.6f}")
            print(f"    Std: {arr.std():.6f}")

            # Find max position
            if arr.ndim == 1:
                max_idx = np.argmax(arr)
                print(f"    Max at index: {max_idx} = {arr[max_idx]:.4f}")
            elif arr.ndim == 2:
                max_idx = np.unravel_index(np.argmax(arr), arr.shape)
                print(f"    Max at index: {max_idx} = {arr[max_idx]:.4f}")

            # Show genomic position if metadata available
            if metadata and 'start' in metadata and 'end' in metadata and arr.shape[0] > 1000:
                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] > 1000):
                    genomic_pos = metadata['start'] + int(max_idx[0] if arr.ndim == 2 else max_idx) * (metadata['end'] - metadata['start']) // arr.shape[0]
                    print(f"    Max genomic position: {metadata.get('chromosome', 'chr?')}:{genomic_pos:,}")
        else:
            print("    EMPTY ARRAY")

    # Interactive exploration
    print("\n" + "="*60)
    print("INTERACTIVE EXPLORATION")
    print("-"*60)

    while True:
        print("\nEnter array number to explore (1-{}) or 'q' to quit:".format(len(data.files)))
        choice = input("> ").strip()

        if choice.lower() == 'q':
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(data.files):
                array_name = data.files[idx]
                explore_array(data[array_name], array_name, metadata)
            else:
                print(f"Invalid choice. Please enter 1-{len(data.files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")

    data.close()

def explore_array(arr, name, metadata=None):
    """Explore a specific array in detail"""
    print("\n" + "="*60)
    print(f"EXPLORING: {name}")
    print("-"*60)
    print(f"Shape: {arr.shape}")
    print(f"Total elements: {arr.size:,}")

    # Helper function to convert index to genomic position
    def idx_to_genomic(idx):
        if metadata and 'start' in metadata and 'end' in metadata:
            if isinstance(idx, (int, np.integer)):
                # For 1D or first dimension of 2D
                genomic_pos = metadata['start'] + idx * (metadata['end'] - metadata['start']) // arr.shape[0]
                return f"{metadata.get('chromosome', 'chr?')}:{genomic_pos:,}"
        return None

    # Helper to format index with genomic position
    def format_position(idx, value=None):
        genomic = idx_to_genomic(idx)
        if genomic:
            if value is not None:
                return f"[{idx:6d}] ({genomic:20s}) = {value:.6f}"
            else:
                return f"[{idx:6d}] ({genomic:20s})"
        else:
            if value is not None:
                return f"[{idx:6d}] = {value:.6f}"
            else:
                return f"[{idx:6d}]"

    if arr.size == 0:
        print("Array is empty!")
        return

    while True:
        print("\nOptions:")
        print("  1. Show statistics (with genomic coords for min/max)")
        print("  2. Show N rows (enter: number/all/range)")
        print("  3. Find peaks (values > threshold)")
        print("  4. Show top N maximum values")
        print("  5. Show histogram of values")
        print("  6. Convert index to genomic position")
        print("  7. Show specific row/column (for 2D arrays)")
        print("  8. Back to main menu")

        choice = input("Select option (1-8): ").strip()

        if choice == '1':
            # Statistics with genomic coordinates
            print("\nStatistics:")
            print(f"  Min: {arr.min():.6f}")
            print(f"  Max: {arr.max():.6f}")
            print(f"  Mean: {arr.mean():.6f}")
            print(f"  Median: {np.median(arr):.6f}")
            print(f"  Std: {arr.std():.6f}")
            print(f"  25th percentile: {np.percentile(arr, 25):.6f}")
            print(f"  75th percentile: {np.percentile(arr, 75):.6f}")
            print(f"  95th percentile: {np.percentile(arr, 95):.6f}")
            print(f"  99th percentile: {np.percentile(arr, 99):.6f}")
            print(f"  99.9th percentile: {np.percentile(arr, 99.9):.6f}")

            # Find min and max positions
            if arr.ndim == 1:
                min_idx = np.argmin(arr)
                max_idx = np.argmax(arr)
                print(f"\nMin location:")
                print(f"  {format_position(min_idx, arr[min_idx])}")
                print(f"\nMax location:")
                print(f"  {format_position(max_idx, arr[max_idx])}")
            elif arr.ndim == 2 and arr.shape[1] == 1:
                # Handle (N, 1) shaped arrays as 1D
                arr_flat = arr.flatten()
                min_idx = np.argmin(arr_flat)
                max_idx = np.argmax(arr_flat)
                print(f"\nMin location:")
                print(f"  {format_position(min_idx, arr_flat[min_idx])}")
                print(f"\nMax location:")
                print(f"  {format_position(max_idx, arr_flat[max_idx])}")
            elif arr.ndim == 2:
                min_idx = np.unravel_index(np.argmin(arr), arr.shape)
                max_idx = np.unravel_index(np.argmax(arr), arr.shape)
                print(f"\nMin location:")
                print(f"  Index: {min_idx}")
                print(f"  Value: {arr[min_idx]:.6f}")
                print(f"\nMax location:")
                print(f"  Index: {max_idx}")
                print(f"  Value: {arr[max_idx]:.6f}")

        elif choice == '2':
            # Show N rows with flexible input
            user_input = input("Enter number of rows, 'all', or range (e.g., 100-200) [10]: ").strip()

            if not user_input:
                user_input = "10"

            if user_input.lower() == 'all':
                if arr.ndim == 1:
                    if arr.size > 1000:
                        confirm = input(f"This will show {arr.size} values. Continue? (y/n): ").strip()
                        if confirm.lower() != 'y':
                            continue
                    print("\nAll values:")
                    for i in range(len(arr)):
                        print(f"  {format_position(i, arr[i])}")
                        if i > 0 and i % 100 == 0 and i < len(arr) - 1:
                            cont = input("Press Enter to continue, 'q' to stop: ").strip()
                            if cont.lower() == 'q':
                                break
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    # Handle (N, 1) shaped arrays as 1D
                    arr_flat = arr.flatten()
                    if arr_flat.size > 1000:
                        confirm = input(f"This will show {arr_flat.size} values. Continue? (y/n): ").strip()
                        if confirm.lower() != 'y':
                            continue
                    print("\nAll values:")
                    for i in range(len(arr_flat)):
                        print(f"  {format_position(i, arr_flat[i])}")
                        if i > 0 and i % 100 == 0 and i < len(arr_flat) - 1:
                            cont = input("Press Enter to continue, 'q' to stop: ").strip()
                            if cont.lower() == 'q':
                                break
                elif arr.ndim == 2:
                    if arr.shape[0] > 100:
                        confirm = input(f"This will show {arr.shape[0]} rows. Continue? (y/n): ").strip()
                        if confirm.lower() != 'y':
                            continue
                    print("\nAll rows:")
                    for i in range(arr.shape[0]):
                        print(f"  Row {i}: {arr[i, :]}")
                        if i > 0 and i % 20 == 0 and i < arr.shape[0] - 1:
                            cont = input("Press Enter to continue, 'q' to stop: ").strip()
                            if cont.lower() == 'q':
                                break

            elif '-' in user_input:
                # Range input
                coord_type = input("Use genomic coordinates (g) or array indices (i)? [i]: ").strip().lower()
                parts = user_input.split('-')

                if coord_type == 'g' and metadata and 'start' in metadata:
                    # Convert genomic to indices
                    start_genomic = int(parts[0].replace(',', ''))
                    end_genomic = int(parts[1].replace(',', ''))
                    start_idx = int((start_genomic - metadata['start']) * arr.shape[0] / (metadata['end'] - metadata['start']))
                    end_idx = int((end_genomic - metadata['start']) * arr.shape[0] / (metadata['end'] - metadata['start']))
                    print(f"\nGenomic range {metadata.get('chromosome')}:{start_genomic:,}-{end_genomic:,}")
                    print(f"Array indices: {start_idx}-{end_idx}")
                else:
                    # Use as array indices
                    start_idx = int(parts[0])
                    end_idx = int(parts[1])

                # Ensure valid range
                start_idx = max(0, start_idx)
                if arr.ndim == 1:
                    end_idx = min(len(arr), end_idx)
                    print(f"\nShowing range [{start_idx}:{end_idx}]:")
                    for i in range(start_idx, end_idx):
                        print(f"  {format_position(i, arr[i])}")
                        if (i - start_idx) > 0 and (i - start_idx) % 50 == 0:
                            cont = input("Press Enter to continue, 'q' to stop: ").strip()
                            if cont.lower() == 'q':
                                break
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    # Handle (N, 1) shaped arrays as 1D
                    arr_flat = arr.flatten()
                    end_idx = min(len(arr_flat), end_idx)
                    print(f"\nShowing range [{start_idx}:{end_idx}]:")
                    for i in range(start_idx, end_idx):
                        print(f"  {format_position(i, arr_flat[i])}")
                        if (i - start_idx) > 0 and (i - start_idx) % 50 == 0:
                            cont = input("Press Enter to continue, 'q' to stop: ").strip()
                            if cont.lower() == 'q':
                                break
                elif arr.ndim == 2:
                    end_idx = min(arr.shape[0], end_idx)
                    print(f"\nShowing rows {start_idx}-{end_idx}:")
                    for i in range(start_idx, end_idx):
                        genomic = idx_to_genomic(i)
                        if genomic:
                            print(f"  Row {i} ({genomic}): {arr[i, :]}")
                        else:
                            print(f"  Row {i}: {arr[i, :]}")

            else:
                # Number of rows
                try:
                    n = int(user_input)
                    if arr.ndim == 1:
                        print(f"\nFirst {n} values:")
                        for i in range(min(n, len(arr))):
                            print(f"  {format_position(i, arr[i])}")
                    elif arr.ndim == 2 and arr.shape[1] == 1:
                        # Handle (N, 1) shaped arrays as 1D
                        arr_flat = arr.flatten()
                        print(f"\nFirst {n} values:")
                        for i in range(min(n, len(arr_flat))):
                            print(f"  {format_position(i, arr_flat[i])}")
                    elif arr.ndim == 2:
                        print(f"\nFirst {n} rows:")
                        for i in range(min(n, arr.shape[0])):
                            genomic = idx_to_genomic(i)
                            if arr.shape[1] > 10:
                                if genomic:
                                    print(f"  Row {i} ({genomic}): {arr[i, :10]}... (showing first 10 of {arr.shape[1]} cols)")
                                else:
                                    print(f"  Row {i}: {arr[i, :10]}... (showing first 10 of {arr.shape[1]} cols)")
                            else:
                                if genomic:
                                    print(f"  Row {i} ({genomic}): {arr[i, :]}")
                                else:
                                    print(f"  Row {i}: {arr[i, :]}")
                except ValueError:
                    print("Invalid input. Please enter a number, 'all', or a range.")

        elif choice == '3':
            # Find peaks
            threshold = input("Enter threshold value: ").strip()
            try:
                threshold = float(threshold)

                if arr.ndim == 1:
                    above = np.where(arr > threshold)[0]
                    count = len(above)
                    print(f"\nValues > {threshold}: {count:,} ({100*count/arr.size:.2f}%)")

                    if count > 0:
                        show_n = input(f"Show how many peaks? [20]: ").strip()
                        show_n = int(show_n) if show_n else 20
                        show_n = min(count, show_n)

                        print(f"\nShowing {show_n} peaks:")
                        for i in range(show_n):
                            idx = above[i]
                            print(f"  {format_position(idx, arr[idx])}")
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    # Handle (N, 1) shaped arrays as 1D
                    arr_flat = arr.flatten()
                    above = np.where(arr_flat > threshold)[0]
                    count = len(above)
                    print(f"\nValues > {threshold}: {count:,} ({100*count/arr_flat.size:.2f}%)")

                    if count > 0:
                        show_n = input(f"Show how many peaks? [20]: ").strip()
                        show_n = int(show_n) if show_n else 20
                        show_n = min(count, show_n)

                        print(f"\nShowing {show_n} peaks:")
                        for i in range(show_n):
                            idx = above[i]
                            print(f"  {format_position(idx, arr_flat[idx])}")
                elif arr.ndim == 2:
                    above = arr > threshold
                    count = np.sum(above)
                    print(f"\nValues > {threshold}: {count:,} ({100*count/arr.size:.2f}%)")

                    if count > 0 and count < 1000:
                        indices = np.where(above)
                        show_n = min(20, len(indices[0]))
                        print(f"\nShowing first {show_n} peaks:")
                        for i in range(show_n):
                            row, col = indices[0][i], indices[1][i]
                            genomic = idx_to_genomic(row)
                            if genomic:
                                print(f"  [{row:4d},{col:2d}] ({genomic}) = {arr[row, col]:.6f}")
                            else:
                                print(f"  [{row:4d},{col:2d}] = {arr[row, col]:.6f}")
            except ValueError:
                print("Invalid threshold value")

        elif choice == '4':
            # Show top N maximum values
            n = input("How many max values to show? [20]: ").strip()
            n = int(n) if n else 20

            if arr.ndim == 1:
                # Get indices of top N values
                if n >= arr.size:
                    top_indices = np.argsort(arr)[::-1]
                else:
                    top_indices = np.argpartition(arr, -n)[-n:]
                    top_indices = top_indices[np.argsort(arr[top_indices])[::-1]]

                print(f"\nTop {min(n, arr.size)} maximum values:")
                for rank, idx in enumerate(top_indices[:n], 1):
                    print(f"  {rank:3d}. {format_position(idx, arr[idx])}")
            elif arr.ndim == 2 and arr.shape[1] == 1:
                # Handle (N, 1) shaped arrays as 1D
                arr_flat = arr.flatten()
                if n >= arr_flat.size:
                    top_indices = np.argsort(arr_flat)[::-1]
                else:
                    top_indices = np.argpartition(arr_flat, -n)[-n:]
                    top_indices = top_indices[np.argsort(arr_flat[top_indices])[::-1]]

                print(f"\nTop {min(n, arr_flat.size)} maximum values:")
                for rank, idx in enumerate(top_indices[:n], 1):
                    print(f"  {rank:3d}. {format_position(idx, arr_flat[idx])}")
            elif arr.ndim == 2:
                # Flatten and find top values
                flat = arr.flatten()
                if n >= flat.size:
                    top_indices = np.argsort(flat)[::-1]
                else:
                    top_indices = np.argpartition(flat, -n)[-n:]
                    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

                print(f"\nTop {min(n, flat.size)} maximum values:")
                for rank, flat_idx in enumerate(top_indices[:n], 1):
                    idx = np.unravel_index(flat_idx, arr.shape)
                    row_genomic = idx_to_genomic(idx[0])
                    if row_genomic:
                        print(f"  {rank:3d}. [{idx[0]:4d},{idx[1]:2d}] ({row_genomic}) = {arr[idx]:.6f}")
                    else:
                        print(f"  {rank:3d}. [{idx[0]:4d},{idx[1]:2d}] = {arr[idx]:.6f}")

        elif choice == '5':
            # Histogram
            bins = input("Number of bins [10]: ").strip()
            bins = int(bins) if bins else 10

            print(f"\nValue distribution ({bins} bins):")
            hist, bin_edges = np.histogram(arr, bins=bins)
            max_count = hist.max()

            for i in range(len(hist)):
                bar_len = int(hist[i] * 50 / max_count) if max_count > 0 else 0
                bar = '*' * bar_len
                print(f"  [{bin_edges[i]:8.3f} - {bin_edges[i+1]:8.3f}]: {bar} ({hist[i]:,})")

        elif choice == '6':
            # Convert index to genomic position
            if metadata and 'start' in metadata:
                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
                    idx = input("Enter array index: ").strip()
                    try:
                        idx = int(idx)
                        if arr.ndim == 1:
                            max_idx = len(arr)
                            val = arr[idx]
                        else:
                            max_idx = arr.shape[0]
                            val = arr[idx, 0]

                        if 0 <= idx < max_idx:
                            genomic_pos = metadata['start'] + idx * (metadata['end'] - metadata['start']) // arr.shape[0]
                            print(f"\nIndex {idx}:")
                            print(f"  Genomic position: {metadata.get('chromosome', 'chr?')}:{genomic_pos:,}")
                            print(f"  Value at position: {val:.6f}")
                        else:
                            print(f"Index out of range. Valid range: 0-{max_idx-1}")
                    except ValueError:
                        print("Invalid index")
                elif arr.ndim == 2:
                    idx = input("Enter row,col: ").strip()
                    try:
                        row, col = map(int, idx.split(','))
                        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                            genomic_pos = metadata['start'] + row * (metadata['end'] - metadata['start']) // arr.shape[0]
                            print(f"\nIndex [{row},{col}]:")
                            print(f"  Genomic position (row): {metadata.get('chromosome', 'chr?')}:{genomic_pos:,}")
                            print(f"  Value at position: {arr[row, col]:.6f}")
                        else:
                            print(f"Index out of range. Valid: row 0-{arr.shape[0]-1}, col 0-{arr.shape[1]-1}")
                    except (ValueError, IndexError):
                        print("Invalid index. Use format: row,col")
            else:
                print("No genomic metadata available")

        elif choice == '7':
            # Show specific row/column for 2D
            if arr.ndim == 2:
                axis = input("Show row (r) or column (c)? ").strip().lower()
                idx = input("Which index? ").strip()

                try:
                    idx = int(idx)
                    if axis == 'r':
                        if 0 <= idx < arr.shape[0]:
                            genomic = idx_to_genomic(idx)
                            if genomic:
                                print(f"\nRow {idx} ({genomic}):")
                            else:
                                print(f"\nRow {idx}:")
                            if arr.shape[1] <= 20:
                                for j in range(arr.shape[1]):
                                    print(f"  Col {j}: {arr[idx, j]:.6f}")
                            else:
                                for j in range(20):
                                    print(f"  Col {j}: {arr[idx, j]:.6f}")
                                print(f"  ... (showing first 20 of {arr.shape[1]} columns)")
                        else:
                            print(f"Row index out of range. Valid: 0-{arr.shape[0]-1}")
                    elif axis == 'c':
                        if 0 <= idx < arr.shape[1]:
                            print(f"\nColumn {idx}:")
                            if arr.shape[0] <= 50:
                                for i in range(arr.shape[0]):
                                    genomic = idx_to_genomic(i)
                                    if genomic:
                                        print(f"  Row {i} ({genomic}): {arr[i, idx]:.6f}")
                                    else:
                                        print(f"  Row {i}: {arr[i, idx]:.6f}")
                            else:
                                for i in range(50):
                                    genomic = idx_to_genomic(i)
                                    if genomic:
                                        print(f"  Row {i} ({genomic}): {arr[i, idx]:.6f}")
                                    else:
                                        print(f"  Row {i}: {arr[i, idx]:.6f}")
                                print(f"  ... (showing first 50 of {arr.shape[0]} rows)")
                        else:
                            print(f"Column index out of range. Valid: 0-{arr.shape[1]-1}")
                except ValueError:
                    print("Invalid index")
            else:
                print("This option is only for 2D arrays")

        elif choice == '8':
            break

        else:
            print("Invalid choice. Please select 1-8.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npz_viewer.py file.npz [metadata.json]")
        print("\nExample:")
        print("  python npz_viewer.py results/IL1B_analysis/IL1B_predict.npz")
        sys.exit(1)

    npz_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        view_npz(npz_file, json_file)
    except FileNotFoundError:
        print(f"Error: File '{npz_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)