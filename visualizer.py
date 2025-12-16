#!/usr/bin/env python3
"""
Advanced visualization system for AlphaGenome NPZ data
Standalone visualizer with comprehensive parameters for publication-quality figures
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
from scipy.ndimage import zoom
from scipy.signal import find_peaks
import sys


class AlphaGenomeVisualizer:
    """Advanced visualization system for AlphaGenome data"""

    # Track display names
    TRACK_NAMES = {
        'dnase': 'DNase/Accessibility',
        'rna_seq': 'RNA-seq',
        'rna_seq_plus': 'RNA-seq (+)',
        'rna_seq_minus': 'RNA-seq (-)',
        'chip_tf': 'TF Binding',
        'chip_histone': 'Histone Marks',
        'h3k4me3': 'H3K4me3',
        'h3k4me1': 'H3K4me1',
        'h3k27ac': 'H3K27ac',
        'h3k27me3': 'H3K27me3',
        'h3k9me3': 'H3K9me3',
        'cage': 'CAGE',
        'procap': 'ProCap'
    }

    # Color schemes
    COLOR_SCHEMES = {
        'default': {
            'dnase': 'green',
            'rna_seq_plus': 'darkblue',
            'rna_seq_minus': 'darkred',
            'chip_tf': 'purple',
            'h3k4me3': 'red',
            'h3k4me1': 'orange',
            'h3k27ac': 'darkgreen',
            'h3k27me3': 'gray',
            'h3k9me3': 'black',
            'cage': 'teal',
            'procap': 'brown'
        },
        'publication': {
            'dnase': '#2E7D32',
            'rna_seq_plus': '#1565C0',
            'rna_seq_minus': '#C62828',
            'chip_tf': '#6A1B9A',
            'h3k4me3': '#D32F2F',
            'h3k4me1': '#F57C00',
            'h3k27ac': '#388E3C',
            'h3k27me3': '#616161',
            'h3k9me3': '#212121',
            'cage': '#00796B',
            'procap': '#5D4037'
        },
        'grayscale': {k: f'C{i}' for i, k in enumerate(TRACK_NAMES.keys())}
    }

    def __init__(self, npz_file, json_file=None):
        """Initialize with AlphaGenome output files"""
        self.npz_path = Path(npz_file)
        self.data = np.load(npz_file)

        # Load metadata if available
        if json_file:
            with open(json_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Try to find matching JSON
            json_path = self.npz_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}

        self.parse_metadata()

    def parse_metadata(self):
        """Extract useful info from metadata"""
        self.chrom = self.metadata.get('chromosome', 'chr?')
        self.start = self.metadata.get('start', 0)
        self.end = self.metadata.get('end', 0)

        # Check for ISM data
        self.is_ism = any('mutation_' in key for key in self.data.keys())

        # Initialize defaults
        self.gene_start = None
        self.gene_end = None
        self.gene_name = ''
        self.tss = None

        # Gene annotation if available (from gene_info field - new format)
        gene_info = self.metadata.get('gene_info', {})
        if gene_info:
            # Direct gene coordinates
            self.gene_start = gene_info.get('start')
            self.gene_end = gene_info.get('end')
            self.gene_name = self.metadata.get('gene', '')  # From top-level gene field
            # TSS is typically at gene start for + strand, end for - strand
            strand = gene_info.get('strand', 1)
            if strand > 0:
                self.tss = self.gene_start
            else:
                self.tss = self.gene_end
        else:
            # Fallback to old format
            gene_annotation = self.metadata.get('gene_annotation', {})
            if gene_annotation:
                self.gene_start = gene_annotation.get('gene_position_in_context', {}).get('start')
                self.gene_end = gene_annotation.get('gene_position_in_context', {}).get('end')
                self.gene_name = gene_annotation.get('symbol', '')
                # Try to get TSS from gene_start as fallback
                self.tss = self.gene_start

    def _add_overlays(self, ax, overlays, show_legend=True, data_len=None):
        """Add overlay regions to a plot based on metadata

        Note: The plot x-axis uses array indices (0 to data_len), not genomic coordinates.
        We need to convert genomic positions to array indices.
        """
        added_labels = set()  # Track which labels we've added

        # Get data length if not provided
        if data_len is None:
            # Try to get from the first data track
            if 'dnase' in self.data:
                data_len = len(self.data['dnase'])
            elif 'rna_seq' in self.data:
                data_len = len(self.data['rna_seq'])
            else:
                return  # Can't determine data length

        # Helper function to convert genomic coordinate to array index
        def genomic_to_index(genomic_pos):
            """Convert genomic position to array index for the current view"""
            # Check if we're zoomed
            if hasattr(self, 'zoom_genomic_start') and hasattr(self, 'zoom_genomic_end'):
                # Map genomic position to zoomed window
                if genomic_pos < self.zoom_genomic_start or genomic_pos > self.zoom_genomic_end:
                    # Position is outside zoom window
                    return None
                # Map to plot coordinates (0 to data_len)
                relative_pos = (genomic_pos - self.zoom_genomic_start) / (self.zoom_genomic_end - self.zoom_genomic_start)
                return relative_pos * data_len
            else:
                # Full range - map to full data
                if self.start and self.end:
                    relative_pos = (genomic_pos - self.start) / (self.end - self.start)
                    return relative_pos * data_len
            return genomic_pos

        for overlay in overlays:
            if overlay == 'tss' and self.tss:
                # TSS line - convert genomic position to array index
                tss_idx = genomic_to_index(self.tss)
                if tss_idx is not None:
                    label = 'TSS' if 'TSS' not in added_labels and show_legend else None
                    ax.axvline(x=tss_idx, color='red', linestyle='--', linewidth=2, alpha=0.8,
                              label=label, zorder=10)
                    if label:
                        added_labels.add('TSS')

            elif overlay == 'gene' and self.gene_start and self.gene_end:
                # Gene body - convert genomic positions to array indices
                gene_start_idx = genomic_to_index(self.gene_start)
                gene_end_idx = genomic_to_index(self.gene_end)
                # Only plot if at least part of gene is visible
                if gene_start_idx is not None or gene_end_idx is not None:
                    # Clip to visible range
                    gene_start_idx = max(0, gene_start_idx) if gene_start_idx is not None else 0
                    gene_end_idx = min(data_len, gene_end_idx) if gene_end_idx is not None else data_len
                    label = f'{self.gene_name or "Gene"}' if self.gene_name not in added_labels and show_legend else None
                    ax.axvspan(gene_start_idx, gene_end_idx, alpha=0.15, color='green',
                              label=label)
                    if label:
                        added_labels.add(self.gene_name or 'Gene')

            elif overlay == 'promoter' and self.tss:
                # Proximal promoter (TSS -500 to -100) - convert to indices
                # Draw this first so core promoter can overlay it
                prox_start_idx = genomic_to_index(self.tss - 500)
                prox_end_idx = genomic_to_index(self.tss - 100)

                # Ensure minimum visible width (at least 0.5% of plot width)
                min_width = data_len * 0.005
                if abs(prox_end_idx - prox_start_idx) < min_width:
                    center = (prox_start_idx + prox_end_idx) / 2
                    prox_start_idx = center - min_width / 2
                    prox_end_idx = center + min_width / 2

                label = 'Proximal Promoter' if 'Proximal Promoter' not in added_labels and show_legend else None
                ax.axvspan(prox_start_idx, prox_end_idx, alpha=0.4, facecolor='orange',
                          label=label, edgecolor='darkorange', linewidth=1, zorder=3)
                if label:
                    added_labels.add('Proximal Promoter')

                # Core promoter (TSS ± 100bp) - convert to indices
                # Draw on top of proximal promoter
                core_start_idx = genomic_to_index(self.tss - 100)
                core_end_idx = genomic_to_index(self.tss + 100)

                if abs(core_end_idx - core_start_idx) < min_width:
                    center = (core_start_idx + core_end_idx) / 2
                    core_start_idx = center - min_width / 2
                    core_end_idx = center + min_width / 2

                label = 'Core Promoter' if 'Core Promoter' not in added_labels and show_legend else None
                # Use pink color to differentiate from TSS red line
                ax.axvspan(core_start_idx, core_end_idx, alpha=0.5, facecolor='pink',
                          label=label, edgecolor='red', linewidth=2, zorder=4)
                if label:
                    added_labels.add('Core Promoter')

            elif overlay == 'enhancer' and self.tss:
                # Upstream enhancer region (-5kb to -2kb from TSS)
                enhancer_start = max(self.start, self.tss - 5000)
                enhancer_end = self.tss - 2000
                if enhancer_start < enhancer_end and enhancer_end > self.start:
                    enhancer_start_idx = genomic_to_index(enhancer_start)
                    enhancer_end_idx = genomic_to_index(enhancer_end)
                    label = 'Enhancer Region' if 'Enhancer Region' not in added_labels and show_legend else None
                    ax.axvspan(enhancer_start_idx, enhancer_end_idx,
                              alpha=0.2, color='purple', label=label)
                    if label:
                        added_labels.add('Enhancer Region')

    def add_regulatory_regions(self, ax, tss, gene_start=None, gene_end=None, targets=None, show_legend=True):
        """Add regulatory region overlays to a plot"""
        if tss:
            # Core promoter (TSS ± 100bp)
            ax.axvspan(tss - 100, tss + 100, alpha=0.2, color='red', label='Core Promoter' if show_legend else None)

            # Proximal promoter (TSS -500 to -100)
            ax.axvspan(tss - 500, tss - 100, alpha=0.2, color='orange', label='Proximal Promoter' if show_legend else None)

            # Upstream enhancer (-5kb to -2kb)
            if tss - 5000 >= self.start:
                ax.axvspan(max(self.start, tss - 5000), tss - 2000, alpha=0.2, color='purple',
                          label='Upstream Enhancer' if show_legend else None)

            # TSS line
            ax.axvline(x=tss, color='red', linestyle='--', linewidth=2, alpha=0.8,
                      label='TSS' if show_legend else None, zorder=10)

        # Gene body
        if gene_start and gene_end:
            ax.axvspan(gene_start, gene_end, alpha=0.1, color='green',
                      label='Gene Body' if show_legend else None)

        # CRISPR targets
        if targets:
            for i, target in enumerate(targets):
                ax.axvline(x=target, color='darkred', linestyle=':', linewidth=2, alpha=0.8,
                          label=f'Target {i+1}' if show_legend else None, zorder=10)

    def parse_track_spec(self, track_spec):
        """
        Parse track specification string
        Examples:
            'all' -> All available tracks
            'rna-seq' -> RNA-seq both strands combined (default)
            'rna-seq:+' -> RNA-seq plus strand only
            'rna-seq:-' -> RNA-seq minus strand only
            'rna-seq:+-' -> RNA-seq both strands separated (same as default)
            'histone:h3k4me3,h3k27ac' -> Specific histone marks
            'histone:all' -> All histone marks
        """
        # Handle special 'all' parameter
        if track_spec == 'all':
            return 'all', {}

        # Support both hyphen and underscore for backwards compatibility
        track_spec = track_spec.replace('rna_seq', 'rna-seq')
        track_spec = track_spec.replace('chip_tf', 'chip-tf')

        parts = track_spec.split(':')
        track_name = parts[0]

        options = {}
        if len(parts) > 1:
            options_str = parts[1]

            # RNA strand options
            if track_name == 'rna-seq':
                if options_str == '+':
                    options['strand'] = 'plus'
                elif options_str == '-':
                    options['strand'] = 'minus'
                elif options_str == '+-' or options_str == '-+':
                    options['strand'] = 'both'

            # Histone mark selection
            elif track_name == 'histone':
                if options_str == 'all':
                    options['marks'] = 'all'
                else:
                    options['marks'] = options_str.split(',')
        else:
            # Default behavior for rna-seq with no options: show both strands
            if track_name == 'rna-seq':
                options['strand'] = 'both'

        return track_name, options

    def get_track_data(self, track_name, options=None):
        """Get data for a specific track with options"""
        options = options or {}

        # Handle 'all' tracks
        if track_name == 'all':
            all_tracks = []
            # Add all available tracks in order
            if 'dnase' in self.data:
                all_tracks.extend(self.get_track_data('dnase'))
            if 'rna_seq' in self.data:
                all_tracks.extend(self.get_track_data('rna-seq', {'strand': 'both'}))
            if 'chip_tf' in self.data:
                all_tracks.extend(self.get_track_data('chip-tf'))
            if 'chip_histone' in self.data:
                all_tracks.extend(self.get_track_data('histone', {'marks': 'all'}))
            if 'cage' in self.data and self.data['cage'].shape[1] > 0:
                all_tracks.extend(self.get_track_data('cage'))
            if 'procap' in self.data and self.data['procap'].shape[1] > 0:
                all_tracks.extend(self.get_track_data('procap'))
            return all_tracks

        if track_name == 'dnase':
            return [('DNase', self.data['dnase'][:, 0])]

        elif track_name == 'rna-seq':
            strand = options.get('strand', 'both')  # Default to both strands
            if strand == 'plus':
                return [('RNA-seq (+)', self.data['rna_seq'][:, 0])]
            elif strand == 'minus':
                return [('RNA-seq (-)', self.data['rna_seq'][:, 1])]
            elif strand == 'both':
                # Return as a single combined track with special marker
                return [('RNA-seq (both strands)', {'plus': self.data['rna_seq'][:, 0],
                                                     'minus': self.data['rna_seq'][:, 1]})]

        elif track_name == 'chip-tf' or track_name == 'chip_tf':
            chip_data = self.data['chip_tf']
            # Upsample to match resolution
            zoom_factor = self.data['dnase'].shape[0] / chip_data.shape[0]
            upsampled = zoom(chip_data[:, 0], zoom_factor)
            return [('TF Binding', upsampled)]

        elif track_name == 'histone':
            histone_data = self.data['chip_histone']
            zoom_factor = self.data['dnase'].shape[0] / histone_data.shape[0]

            marks = options.get('marks', 'all')
            histone_names = ['H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K27me3', 'H3K9me3']

            tracks = []
            if marks == 'all':
                for i, name in enumerate(histone_names[:histone_data.shape[1]]):
                    upsampled = zoom(histone_data[:, i], zoom_factor)
                    tracks.append((name, upsampled))
            else:
                for mark in marks:
                    if mark.lower() in [h.lower() for h in histone_names]:
                        idx = [h.lower() for h in histone_names].index(mark.lower())
                        if idx < histone_data.shape[1]:
                            upsampled = zoom(histone_data[:, idx], zoom_factor)
                            tracks.append((histone_names[idx], upsampled))

            return tracks

        elif track_name == 'cage':
            if self.data['cage'].shape[1] > 0:
                return [('CAGE', self.data['cage'][:, 0])]

        elif track_name == 'procap':
            if self.data['procap'].shape[1] > 0:
                return [('ProCap', self.data['procap'][:, 0])]

        elif track_name == 'score' or track_name == 'scorer':
            # Handle score data (from score_interval method)
            scorer_keys = [k for k in self.data.keys() if k.startswith('scorer')]
            if scorer_keys:
                # Use first scorer data
                scorer_data = self.data[scorer_keys[0]]
                return [('Score', scorer_data)]

        elif track_name == 'ism':
            # Handle ISM data - compute L2 norms and create heatmap
            return self._process_ism_data()

        return []

    def visualize(self, tracks, canvas='single', style='default',
                 figsize=None, dpi=150, output=None,
                 highlight_gene=True, zoom_region=None,
                 normalize=False, smooth=None, peaks=False,
                 title=None, **kwargs):
        """
        Main visualization method

        Args:
            tracks: List of track specifications
            canvas: 'single', 'multi', or 'grid'
            style: 'default', 'publication', 'grayscale'
            figsize: Custom figure size (width, height)
            dpi: DPI for output
            output: Output filename
            highlight_gene: Show gene annotation
            zoom_region: (start, end) to zoom to specific region
            normalize: Normalize each track to [0, 1]
            smooth: Smoothing window size
            peaks: Show peak calls
            title: Custom title
        """

        # Parse all track specifications
        all_track_data = []
        for track_spec in tracks:
            track_name, options = self.parse_track_spec(track_spec)
            track_data = self.get_track_data(track_name, options)
            all_track_data.extend(track_data)

        if not all_track_data:
            print("No valid tracks to visualize")
            return

        # Determine canvas layout
        if canvas == 'single':
            self._plot_single_canvas(all_track_data, style, figsize, dpi,
                                    output, highlight_gene, zoom_region,
                                    normalize, smooth, peaks, title, **kwargs)
        elif canvas == 'multi':
            self._plot_multi_canvas(all_track_data, style, figsize, dpi,
                                  output, highlight_gene, zoom_region,
                                  normalize, smooth, peaks, title, **kwargs)
        elif canvas == 'grid':
            self._plot_grid_canvas(all_track_data, style, figsize, dpi,
                                 output, highlight_gene, zoom_region,
                                 normalize, smooth, peaks, title, **kwargs)

    def _process_ism_data(self):
        """Process ISM mutation data using metadata and compute impacts"""
        import numpy as np

        # Check if we have metadata with mutations list
        if not self.metadata or 'mutations' not in self.metadata:
            # Fallback: compute from raw data
            return self._compute_ism_from_raw()

        # Use metadata approach (from working archive code)
        mutation_impacts = []
        impact_matrix = {}  # For heatmap: pos -> {base -> impact}

        for mut in self.metadata['mutations']:
            mut_idx = mut['mutation_idx']
            scorer_impacts = []

            # Compute impact for each scorer
            for scorer_idx in range(20):
                key = f'mutation_{mut_idx}_scorer_{scorer_idx}'
                if key in self.data:
                    data = self.data[key]
                    if data.size > 0:
                        # Simple L2 norm of the data itself
                        impact = np.sqrt(np.sum(data**2))
                        if not np.isnan(impact):
                            scorer_impacts.append(impact)

            # Take mean of scorer impacts
            if scorer_impacts:
                mean_impact = np.mean(scorer_impacts)
                mutation_impacts.append(mean_impact)

                # Parse variant for heatmap
                variant = mut['variant']
                parts = variant.split(':')
                if len(parts) >= 3:
                    pos = int(parts[1])
                    mutation = parts[2]
                    if '>' in mutation:
                        ref, alt = mutation.split('>')
                        if pos not in impact_matrix:
                            impact_matrix[pos] = {}
                        impact_matrix[pos][alt] = mean_impact
            else:
                mutation_impacts.append(0.0)

        # Store for heatmap use
        self.impact_matrix = impact_matrix
        self.mutation_impacts = np.array(mutation_impacts)

        print(f"Processed {len(mutation_impacts)} mutations from metadata")
        if mutation_impacts:
            impacts_array = np.array(mutation_impacts)
            if not np.all(np.isnan(impacts_array)):
                print(f"Max impact: {np.nanmax(impacts_array):.1f}, Mean: {np.nanmean(impacts_array):.1f}")
            return [('ISM Impact', impacts_array)]
        return []

    def _compute_ism_from_raw(self):
        """Fallback: compute from raw NPZ without metadata"""
        import numpy as np

        mutations = set()
        for key in self.data.keys():
            if key.startswith('mutation_'):
                mut_idx = int(key.split('_')[1])
                mutations.add(mut_idx)

        ism_impacts = []
        for mut_idx in sorted(mutations):
            if mut_idx == 0:
                continue

            scorer_impacts = []
            for scorer_idx in range(20):
                key = f'mutation_{mut_idx}_scorer_{scorer_idx}'
                if key in self.data:
                    data = self.data[key]
                    if data.size > 0:
                        impact = np.sqrt(np.sum(data**2))
                        if not np.isnan(impact):
                            scorer_impacts.append(impact)

            if scorer_impacts:
                ism_impacts.append(np.mean(scorer_impacts))
            else:
                ism_impacts.append(0.0)

        return [('ISM Impact', np.array(ism_impacts))] if ism_impacts else []

    def _plot_ism_heatmap(self, ism_data, ax, title="ISM Impact Heatmap"):
        """Plot ISM data as proper heatmap using impact matrix"""
        import numpy as np
        import matplotlib.pyplot as plt

        impacts = ism_data

        # Check if we have impact_matrix for proper heatmap
        if hasattr(self, 'impact_matrix') and self.impact_matrix:
            # Create proper heatmap from impact_matrix
            positions = sorted(self.impact_matrix.keys())
            bases = ['A', 'C', 'G', 'T']

            # Build heatmap data
            heatmap_data = []
            for base in bases:
                row = []
                for pos in positions:
                    if pos in self.impact_matrix and base in self.impact_matrix[pos]:
                        row.append(self.impact_matrix[pos][base])
                    else:
                        row.append(0)
                heatmap_data.append(row)

            heatmap_array = np.array(heatmap_data)

            # Create heatmap
            im = ax.imshow(heatmap_array, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_yticks(range(4))
            ax.set_yticklabels(bases)
            ax.set_ylabel('Alt Base', fontsize=11)
            ax.set_xlabel('Genomic Position', fontsize=11)

            # Set x-labels (show subset if too many)
            if len(positions) <= 30:
                ax.set_xticks(range(len(positions)))
                ax.set_xticklabels([str(p)[-4:] for p in positions], rotation=45)
            else:
                step = len(positions) // 10
                ax.set_xticks(range(0, len(positions), step))
                ax.set_xticklabels([str(positions[i])[-4:] for i in range(0, len(positions), step)], rotation=45)

            plt.colorbar(im, ax=ax, pad=0.02, label='ISM Impact')

            # Add title with genomic coordinates
            if self.metadata and 'ism_start' in self.metadata:
                start = self.metadata['ism_start']
                end = self.metadata['ism_end']
                chrom = self.metadata.get('chromosome', 'chr?')
                ax.set_title(f'{title}\n{chrom}:{start}-{end}', fontsize=13, fontweight='bold')
            else:
                ax.set_title(title, fontsize=13, fontweight='bold')

            # Mark highest impact
            max_val = np.max(heatmap_array)
            max_pos = np.unravel_index(np.argmax(heatmap_array), heatmap_array.shape)
            from matplotlib.patches import Rectangle
            rect = Rectangle((max_pos[1]-0.45, max_pos[0]-0.45), 0.9, 0.9,
                           fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(rect)

        else:
            # Fallback to bar chart if no impact_matrix
            n_mutations = len(impacts)
            bars = ax.bar(range(n_mutations), impacts, color='darkred', alpha=0.8)
            ax.set_xlabel('Mutation Index', fontsize=12)
            ax.set_ylabel('ISM Impact', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Add stats annotation
        if len(impacts) > 0 and not np.all(np.isnan(impacts)):
            max_val = np.nanmax(impacts)
            ax.text(0.02, 0.98, f'Max: {max_val:.1f}',
                   transform=ax.transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

        # Add annotations
        ax.text(0.02, 0.98, f'Max Impact: {impacts[max_idx]:.1f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.text(0.02, 0.92, f'Total Mutations: {n_mutations}',
                transform=ax.transAxes, fontsize=9, va='top')

        # Add gridlines for clarity
        ax.set_xticks(np.arange(-0.5, seq_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.3)

    def _plot_single_canvas(self, track_data, style, figsize, dpi, output,
                           highlight_gene, zoom_region, normalize, smooth,
                           peaks, title, **kwargs):
        """Plot all tracks on a single axis"""

        figsize = figsize or (15, 6)
        fig, ax = plt.subplots(figsize=figsize)

        colors = self.COLOR_SCHEMES.get(style, self.COLOR_SCHEMES['default'])

        # Determine data length (handle dict for RNA-seq both strands)
        first_data = track_data[0][1]
        if isinstance(first_data, dict):
            data_len = len(first_data['plus'])
        else:
            data_len = len(first_data)

        # Handle zoom if specified
        if zoom_region:
            # zoom_region contains genomic coordinates
            genomic_start = self.metadata['start']
            genomic_end = self.metadata['end']
            genomic_span = genomic_end - genomic_start

            # Convert genomic coordinates to array indices
            self.zoom_x_start = max(0, int((zoom_region[0] - genomic_start) * data_len / genomic_span))
            self.zoom_x_end = min(data_len, int((zoom_region[1] - genomic_start) * data_len / genomic_span))

            # Store zoom genomic coordinates
            self.zoom_genomic_start = zoom_region[0]
            self.zoom_genomic_end = zoom_region[1]

            # Don't downsample when zoomed
            print(f"Zooming to {self.zoom_genomic_start:,}-{self.zoom_genomic_end:,}")
        else:
            # Downsample if data is too large (>100k points)
            if data_len > 100000:
                print(f"Note: Downsampling by {data_len//50000}x for visualization (use --zoom for full resolution)")

        # Check if this is ISM data
        if track_data and track_data[0][0] == 'ISM Impact':
            # Use ISM heatmap visualization
            self._plot_ism_heatmap(track_data[0][1], ax)
            plt.tight_layout()
            if output:
                plt.savefig(output, dpi=dpi, bbox_inches='tight')
                print(f"Saved ISM heatmap to {output}")
            else:
                plt.show()
            return

        offset = 0
        for name, data in track_data:
            # Check if this is score data (2D heatmap)
            if isinstance(data, np.ndarray) and len(data.shape) == 2 and name == 'Score':
                # Clear axis for heatmap
                ax.clear()

                # Create extent for genomic coordinates
                extent = [self.start, self.end, 0, data.shape[0]]
                im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                              interpolation='nearest', extent=extent)

                # Format x-axis with more precision
                from matplotlib.ticker import FuncFormatter, MultipleLocator
                def format_genomic_pos(x, p):
                    # Format with commas for thousands separator
                    return f'{int(x):,}'
                ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_pos))
                # Add more tick marks every 50kb
                ax.xaxis.set_major_locator(MultipleLocator(50000))
                ax.set_xlabel(f'Genomic Position on {self.chrom} (bp)')
                ax.set_ylabel('Score Tracks')

                # Add colorbar
                plt.colorbar(im, ax=ax, label='Score')

                # Mark gene if available
                if highlight_gene and self.gene_start and self.gene_end:
                    # Get gene name from metadata
                    gene_name = self.metadata.get('gene_annotation', {}).get('name', 'Gene')
                    ax.axvline(x=self.gene_start, color='green', linestyle='--',
                              alpha=0.7, linewidth=2, label=f"{gene_name} TSS")
                    ax.axvspan(self.gene_start, self.gene_end,
                              alpha=0.2, color='green', label=f"{gene_name} gene body")
                    ax.legend(loc='upper right')

                ax.set_title(title or f'Score Heatmap: {self.chrom}:{self.start:,}-{self.end:,}')
                continue  # Skip regular plotting

            # Check if this is a both-strands RNA-seq track
            elif isinstance(data, dict) and 'plus' in data and 'minus' in data:
                # Handle both strands in one track
                plus_data = data['plus']
                minus_data = data['minus']

                # Handle zoom or downsampling
                if zoom_region and hasattr(self, 'zoom_x_start'):
                    # Extract zoomed region
                    plus_data = plus_data[self.zoom_x_start:self.zoom_x_end]
                    minus_data = minus_data[self.zoom_x_start:self.zoom_x_end]
                elif not zoom_region and data_len > 100000:
                    # Downsample for large datasets
                    downsample = data_len // 50000
                    plus_data = plus_data[::downsample]
                    minus_data = minus_data[::downsample]

                # Create x array matching data length
                track_x = np.arange(len(plus_data))

                # Apply smoothing
                if smooth:
                    from scipy.ndimage import uniform_filter1d
                    plus_data = uniform_filter1d(plus_data, smooth)
                    minus_data = uniform_filter1d(minus_data, smooth)

                # Normalize if requested
                if normalize:
                    max_val = max(plus_data.max(), minus_data.max())
                    if max_val > 0:
                        plus_data = plus_data / max_val
                        minus_data = minus_data / max_val

                # Plot both strands - plus above, minus below
                ax.fill_between(track_x, offset, offset + plus_data, alpha=0.7,
                               color=colors.get('rna_seq_plus', 'darkblue'), label='RNA (+)')
                ax.fill_between(track_x, offset, offset - minus_data, alpha=0.7,
                               color=colors.get('rna_seq_minus', 'darkred'), label='RNA (-)')
                ax.axhline(y=offset, color='black', linewidth=0.5, alpha=0.3)

                # Adjust offset for next track
                if normalize:
                    offset += 2.5  # More space for both strands
                else:
                    offset += max(plus_data.max(), minus_data.max()) * 1.2

                continue  # Skip the regular plotting below

            # Regular single data track
            # Handle zoom or downsampling
            if zoom_region and hasattr(self, 'zoom_x_start'):
                # Extract zoomed region
                data = data[self.zoom_x_start:self.zoom_x_end]
            elif not zoom_region and data_len > 100000:
                # Downsample for large datasets
                downsample = data_len // 50000
                data = data[::downsample]

            # Create x array matching data length
            track_x = np.arange(len(data))

            # Apply smoothing
            if smooth:
                from scipy.ndimage import uniform_filter1d
                data = uniform_filter1d(data, smooth)

            # Normalize if requested
            if normalize and data.max() > 0:
                data = data / data.max()

            # Determine color
            color_key = name.lower().replace(' ', '_').replace('(+)', 'plus').replace('(-)', 'minus')
            color = colors.get(color_key, 'gray')

            # Plot
            ax.fill_between(track_x, offset, offset + data, alpha=0.7, color=color, label=name)

            # Peak calling
            if peaks and data.max() > 0:
                peak_idx, _ = find_peaks(data, height=data.max() * 0.3, distance=100)
                # Limit number of peaks shown to avoid overflow
                if len(peak_idx) > 100:
                    peak_idx = peak_idx[::len(peak_idx)//100]  # Sample to 100 peaks
                if len(peak_idx) > 0 and len(peak_idx) < 1000:
                    ax.scatter(track_x[peak_idx], offset + data[peak_idx],
                             color='red', s=10, zorder=10, marker='v')

            if normalize:
                offset += 1.2
            else:
                offset += data.max() * 1.1 if data.max() > 0 else 0

        # Add overlays if requested
        overlays = kwargs.get('overlays', [])
        if overlays:
            # Get the actual x-axis limits to determine plotted data length
            xlim = ax.get_xlim()
            actual_data_len = int(xlim[1] - xlim[0])
            if actual_data_len <= 0:
                # Fallback to calculated length
                if hasattr(self, 'zoom_x_start') and hasattr(self, 'zoom_x_end'):
                    actual_data_len = self.zoom_x_end - self.zoom_x_start
                elif data_len > 100000 and not zoom_region:
                    # We downsampled
                    downsample = data_len // 50000
                    actual_data_len = data_len // downsample
                else:
                    actual_data_len = data_len
            self._add_overlays(ax, overlays, show_legend=True, data_len=actual_data_len)
        # Legacy support for regulatory regions
        elif kwargs.get('show_regulatory') or kwargs.get('tss'):
            self.add_regulatory_regions(ax,
                                       kwargs.get('tss'),
                                       kwargs.get('gene_start'),
                                       kwargs.get('gene_end'),
                                       kwargs.get('crispr_targets'),
                                       show_legend=True)
        # Gene highlight (only if not showing other overlays to avoid clutter)
        elif highlight_gene and self.gene_start and self.gene_end:
            ax.axvspan(self.gene_start, self.gene_end, alpha=0.2, color='yellow',
                      label=self.gene_name or 'Gene')

        # Formatting with genomic coordinates
        # Convert x-axis to genomic coordinates
        from matplotlib.ticker import FuncFormatter, MultipleLocator

        def format_genomic_pos(x_val, p):
            # Get the limits of the actual plot
            xlim = ax.get_xlim()
            plot_width = xlim[1] - xlim[0]

            # Determine genomic range
            if hasattr(self, 'zoom_genomic_start') and hasattr(self, 'zoom_genomic_end'):
                # We're zoomed
                genomic_start = self.zoom_genomic_start
                genomic_end = self.zoom_genomic_end
            else:
                # Full range
                genomic_start = self.start
                genomic_end = self.end

            # Map x_val to genomic position
            if plot_width > 0:
                rel_pos = (x_val - xlim[0]) / plot_width
                genomic_pos = genomic_start + rel_pos * (genomic_end - genomic_start)
            else:
                genomic_pos = genomic_start

            # Format with commas for thousands separator
            return f'{int(genomic_pos):,}'

        ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_pos))

        # Let matplotlib handle tick placement automatically
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

        ax.set_xlabel(f'Genomic Position on {self.chrom} (bp)')
        ax.set_ylabel('Signal' if not normalize else 'Normalized Signal')
        # Update title to show range and resolution
        if hasattr(self, 'zoom_genomic_start'):
            resolution = self.zoom_genomic_end - self.zoom_genomic_start
            default_title = f'Regulatory Tracks: {self.chrom}:{self.zoom_genomic_start:,}-{self.zoom_genomic_end:,} (Resolution: {resolution:,} bp)'
        else:
            resolution = self.end - self.start
            default_title = f'Regulatory Tracks: {self.chrom}:{self.start:,}-{self.end:,} (Resolution: {resolution:,} bp)'
        ax.set_title(title or default_title)

        # Add legend after all elements are plotted (including overlays)
        # Force collection of all labeled elements
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        # Adjust layout to prevent legend cutoff
        plt.subplots_adjust(right=0.85)

        # Save
        if output:
            plt.savefig(output, dpi=dpi, bbox_inches='tight')
            print(f"Saved to {output}")
        else:
            plt.show()

    def _plot_multi_canvas(self, track_data, style, figsize, dpi, output,
                          highlight_gene, zoom_region, normalize, smooth,
                          peaks, title, **kwargs):
        """Plot each track on separate subplot"""

        n_tracks = len(track_data)
        figsize = figsize or (15, 3 * n_tracks)
        fig, axes = plt.subplots(n_tracks, 1, figsize=figsize, sharex=True)

        if n_tracks == 1:
            axes = [axes]

        colors = self.COLOR_SCHEMES.get(style, self.COLOR_SCHEMES['default'])

        # Store zoom information if provided
        if zoom_region and self.metadata and 'start' in self.metadata:
            self.zoom_genomic_start = zoom_region[0]
            self.zoom_genomic_end = zoom_region[1]

        for i, (name, data) in enumerate(track_data):
            ax = axes[i]

            # Handle score data (2D heatmap)
            if isinstance(data, np.ndarray) and len(data.shape) == 2 and name == 'Score':
                # Create extent for genomic coordinates
                extent = [self.start, self.end, 0, data.shape[0]]
                im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                              interpolation='nearest', extent=extent)

                # Format x-axis with more precision
                from matplotlib.ticker import FuncFormatter, MultipleLocator
                def format_genomic_pos(x, p):
                    # Format with commas for thousands separator
                    return f'{int(x):,}'
                ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_pos))
                # Add more tick marks every 50kb
                ax.xaxis.set_major_locator(MultipleLocator(50000))
                ax.set_xlabel(f'Genomic Position on {self.chrom} (bp)')
                ax.set_ylabel('Score Tracks')

                # Add colorbar
                plt.colorbar(im, ax=ax, label='Score')

                # Mark gene if available
                if highlight_gene and self.gene_start and self.gene_end:
                    gene_name = self.metadata.get('gene_annotation', {}).get('name', 'Gene')
                    ax.axvline(x=self.gene_start, color='green', linestyle='--',
                              alpha=0.7, linewidth=2)
                    ax.axvspan(self.gene_start, self.gene_end,
                              alpha=0.2, color='green')

                ax.set_title(f'{name}')
                continue

            # Handle both strands RNA-seq
            elif isinstance(data, dict) and 'plus' in data and 'minus' in data:
                plus_data = data['plus']
                minus_data = data['minus']

                # Apply zoom if specified
                if zoom_region:
                    # Check if we have genomic coordinates in metadata
                    if self.metadata and 'start' in self.metadata and 'end' in self.metadata:
                        # Convert genomic coordinates to array indices
                        genomic_start = self.metadata['start']
                        genomic_end = self.metadata['end']
                        data_len = len(plus_data)
                        genomic_span = genomic_end - genomic_start

                        # Calculate array indices from genomic coordinates
                        start_idx = max(0, int((zoom_region[0] - genomic_start) * data_len / genomic_span))
                        end_idx = min(data_len, int((zoom_region[1] - genomic_start) * data_len / genomic_span))

                        if end_idx > start_idx:
                            # Use 0-based indices for x-axis, formatter will handle conversion
                            x = np.arange(end_idx - start_idx)
                            plus_data = plus_data[start_idx:end_idx]
                            minus_data = minus_data[start_idx:end_idx]
                        else:
                            print(f"Warning: Invalid zoom region {zoom_region}")
                            x = np.arange(len(plus_data))
                    else:
                        # Direct array indices
                        x = np.arange(0, min(zoom_region[1], len(plus_data)) - zoom_region[0])
                        plus_data = plus_data[zoom_region[0]:min(zoom_region[1], len(plus_data))]
                        minus_data = minus_data[zoom_region[0]:min(zoom_region[1], len(minus_data))]
                else:
                    x = np.arange(len(plus_data))

                # Apply smoothing
                if smooth:
                    from scipy.ndimage import uniform_filter1d
                    plus_data = uniform_filter1d(plus_data, smooth)
                    minus_data = uniform_filter1d(minus_data, smooth)

                # Normalize if requested
                if normalize:
                    max_val = max(plus_data.max(), minus_data.max())
                    if max_val > 0:
                        plus_data = plus_data / max_val
                        minus_data = minus_data / max_val

                # Plot both strands
                ax.fill_between(x, plus_data, alpha=0.7,
                               color=colors.get('rna_seq_plus', 'darkblue'), label='(+)')
                ax.fill_between(x, -minus_data, alpha=0.7,
                               color=colors.get('rna_seq_minus', 'darkred'), label='(-)')
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
                ax.set_ylabel('RNA-seq (±)')
                ax.legend(loc='upper right', fontsize=8)

                # Add overlays for RNA-seq both strands
                overlays = kwargs.get('overlays', [])
                if overlays:
                    self._add_overlays(ax, overlays, show_legend=(i == 0))
                # Gene highlight (only if not showing other overlays)
                elif highlight_gene and self.gene_start and self.gene_end:
                    ax.axvspan(self.gene_start, self.gene_end, alpha=0.2, color='yellow')

                # Remove x-axis labels except for bottom
                if i < n_tracks - 1:
                    ax.set_xticklabels([])

                continue  # Skip regular processing

            # Regular single track processing
            # Apply zoom if specified
            if zoom_region:
                # Check if we have genomic coordinates in metadata
                if self.metadata and 'start' in self.metadata and 'end' in self.metadata:
                    # Convert genomic coordinates to array indices
                    genomic_start = self.metadata['start']
                    genomic_end = self.metadata['end']
                    data_len = len(data)
                    genomic_span = genomic_end - genomic_start

                    # Calculate array indices from genomic coordinates
                    start_idx = max(0, int((zoom_region[0] - genomic_start) * data_len / genomic_span))
                    end_idx = min(data_len, int((zoom_region[1] - genomic_start) * data_len / genomic_span))

                    if end_idx > start_idx:
                        # Use 0-based indices for x-axis, formatter will handle conversion
                        x = np.arange(end_idx - start_idx)
                        data = data[start_idx:end_idx]
                    else:
                        print(f"Warning: Invalid zoom region {zoom_region}")
                        x = np.arange(len(data))
                else:
                    # Direct array indices
                    x = np.arange(0, min(zoom_region[1], len(data)) - zoom_region[0])
                    data = data[zoom_region[0]:min(zoom_region[1], len(data))]
            else:
                x = np.arange(len(data))

            # Apply smoothing
            if smooth:
                from scipy.ndimage import uniform_filter1d
                data = uniform_filter1d(data, smooth)

            # Normalize if requested
            if normalize and data.max() > 0:
                data = data / data.max()

            # Determine color
            color_key = name.lower().replace(' ', '_').replace('(+)', 'plus').replace('(-)', 'minus')
            color = colors.get(color_key, 'gray')

            # Plot
            ax.fill_between(x, data, alpha=0.7, color=color)
            ax.set_ylabel(name)

            # Peak calling
            if peaks and data.max() > 0:
                peak_idx, _ = find_peaks(data, height=data.max() * 0.3, distance=100)
                # Limit number of peaks shown to avoid overflow
                if len(peak_idx) > 100:
                    peak_idx = peak_idx[::len(peak_idx)//100]  # Sample to 100 peaks
                if len(peak_idx) > 0 and len(peak_idx) < 1000:
                    ax.scatter(x[peak_idx], data[peak_idx],
                             color='red', s=10, zorder=10, marker='v')

            # Add overlays if requested (on each subplot)
            overlays = kwargs.get('overlays', [])
            if overlays:
                self._add_overlays(ax, overlays, show_legend=(i == 0))  # Only legend on first
            # Legacy regulatory regions
            elif kwargs.get('show_regulatory'):
                self.add_regulatory_regions(ax,
                                           kwargs.get('tss'),
                                           kwargs.get('gene_start'),
                                           kwargs.get('gene_end'),
                                           kwargs.get('crispr_targets'),
                                           show_legend=(i == 0))  # Only show legend on first plot
            # Gene highlight (only if not showing other overlays)
            elif highlight_gene and self.gene_start and self.gene_end:
                ax.axvspan(self.gene_start, self.gene_end, alpha=0.2, color='yellow')

            # Remove x-axis labels except for bottom
            if i < n_tracks - 1:
                ax.set_xticklabels([])

        # Set overall labels with genomic coordinates
        from matplotlib.ticker import FuncFormatter, MultipleLocator

        def format_genomic_pos(x_val, p):
            # Get actual data length
            first_track = track_data[0][1] if track_data else []
            if isinstance(first_track, dict):
                data_len = len(first_track.get('plus', []))
            else:
                data_len = len(first_track) if hasattr(first_track, '__len__') else 131072

            # Check if we're in zoom mode
            if hasattr(self, 'zoom_genomic_start') and hasattr(self, 'zoom_genomic_end'):
                # Map x_val to the zoom range
                if zoom_region and self.metadata and 'start' in self.metadata:
                    # Calculate the zoomed data range
                    genomic_start = self.metadata['start']
                    genomic_end = self.metadata['end']
                    genomic_span = genomic_end - genomic_start
                    x_start = max(0, int((self.zoom_genomic_start - genomic_start) * data_len / genomic_span))
                    x_end = min(data_len, int((self.zoom_genomic_end - genomic_start) * data_len / genomic_span))
                    plot_len = x_end - x_start
                else:
                    plot_len = data_len

                if plot_len > 0:
                    rel_pos = x_val / plot_len
                    genomic_pos = self.zoom_genomic_start + rel_pos * (self.zoom_genomic_end - self.zoom_genomic_start)
                else:
                    genomic_pos = self.zoom_genomic_start
            else:
                # Full range mapping
                if data_len > 0:
                    genomic_pos = self.start + (x_val / data_len) * (self.end - self.start)
                else:
                    genomic_pos = self.start + x_val

            # Format with commas for thousands separator
            return f'{int(genomic_pos):,}'

        # Apply formatter to the last axis
        axes[-1].xaxis.set_major_formatter(FuncFormatter(format_genomic_pos))

        axes[-1].set_xlabel(f'Genomic Position on {self.chrom} (bp)')
        # Update title to show range and resolution
        if hasattr(self, 'zoom_genomic_start'):
            resolution = self.zoom_genomic_end - self.zoom_genomic_start
            default_title = f'Regulatory Tracks: {self.chrom}:{self.zoom_genomic_start:,}-{self.zoom_genomic_end:,} (Resolution: {resolution:,} bp)'
        else:
            resolution = self.end - self.start
            default_title = f'Regulatory Tracks: {self.chrom}:{self.start:,}-{self.end:,} (Resolution: {resolution:,} bp)'
        axes[0].set_title(title or default_title)

        plt.tight_layout()

        # Save
        if output:
            plt.savefig(output, dpi=dpi, bbox_inches='tight')
            print(f"Saved to {output}")
        else:
            plt.show()

    def _plot_grid_canvas(self, track_data, style, figsize, dpi, output,
                         highlight_gene, zoom_region, normalize, smooth,
                         peaks, title, **kwargs):
        """Plot tracks in a grid layout"""

        n_tracks = len(track_data)
        cols = kwargs.get('grid_cols', 2)
        rows = (n_tracks + cols - 1) // cols

        figsize = figsize or (7 * cols, 4 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_tracks > 1 else [axes]

        colors = self.COLOR_SCHEMES.get(style, self.COLOR_SCHEMES['default'])

        for i, (name, data) in enumerate(track_data):
            ax = axes[i]

            # Handle score data (2D heatmap)
            if isinstance(data, np.ndarray) and len(data.shape) == 2 and name == 'Score':
                # Create extent for genomic coordinates
                extent = [self.start, self.end, 0, data.shape[0]]
                im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                              interpolation='nearest', extent=extent)

                # Format x-axis with more precision
                from matplotlib.ticker import FuncFormatter, MultipleLocator
                def format_genomic_pos(x, p):
                    # Format with commas for thousands separator
                    return f'{int(x):,}'
                ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_pos))
                ax.set_xlabel(f'Position on {self.chrom} (Mb)')
                ax.set_ylabel('Score Tracks')
                ax.set_title(name)

                # Add colorbar
                plt.colorbar(im, ax=ax)
                continue

            # Handle both strands RNA-seq data
            if isinstance(data, dict) and 'plus' in data and 'minus' in data:
                plus_data = data['plus']
                minus_data = data['minus']

                # Apply zoom if specified
                if zoom_region:
                    # Check if we have genomic coordinates in metadata
                    if self.metadata and 'start' in self.metadata and 'end' in self.metadata:
                        # Convert genomic coordinates to array indices
                        genomic_start = self.metadata['start']
                        genomic_end = self.metadata['end']
                        data_len = len(plus_data)
                        genomic_span = genomic_end - genomic_start

                        # Calculate array indices from genomic coordinates
                        start_idx = max(0, int((zoom_region[0] - genomic_start) * data_len / genomic_span))
                        end_idx = min(data_len, int((zoom_region[1] - genomic_start) * data_len / genomic_span))

                        if end_idx > start_idx:
                            # Use 0-based indices for x-axis, formatter will handle conversion
                            x = np.arange(end_idx - start_idx)
                            plus_data = plus_data[start_idx:end_idx]
                            minus_data = minus_data[start_idx:end_idx]
                        else:
                            print(f"Warning: Invalid zoom region {zoom_region}")
                            x = np.arange(len(plus_data))
                    else:
                        # Direct array indices
                        x = np.arange(0, min(zoom_region[1], len(plus_data)) - zoom_region[0])
                        plus_data = plus_data[zoom_region[0]:min(zoom_region[1], len(plus_data))]
                        minus_data = minus_data[zoom_region[0]:min(zoom_region[1], len(minus_data))]
                else:
                    x = np.arange(len(plus_data))

                # Apply smoothing
                if smooth:
                    from scipy.ndimage import uniform_filter1d
                    plus_data = uniform_filter1d(plus_data, smooth)
                    minus_data = uniform_filter1d(minus_data, smooth)

                # Normalize if requested
                if normalize:
                    max_val = max(plus_data.max(), minus_data.max())
                    if max_val > 0:
                        plus_data = plus_data / max_val
                        minus_data = minus_data / max_val

                # Plot both strands
                ax.fill_between(x, plus_data, alpha=0.7,
                               color=colors.get('rna_seq_plus', 'darkblue'), label='(+)')
                ax.fill_between(x, -minus_data, alpha=0.7,
                               color=colors.get('rna_seq_minus', 'darkred'), label='(-)')
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
                ax.set_title(name)
                ax.set_xlabel('Position')
                ax.set_ylabel('RNA-seq (±)')
                ax.legend(loc='upper right', fontsize=8)
                continue

            # Apply zoom if specified
            if zoom_region:
                # Check if we have genomic coordinates in metadata
                if self.metadata and 'start' in self.metadata and 'end' in self.metadata:
                    # Convert genomic coordinates to array indices
                    genomic_start = self.metadata['start']
                    genomic_end = self.metadata['end']
                    data_len = len(data)
                    genomic_span = genomic_end - genomic_start

                    # Calculate array indices from genomic coordinates
                    start_idx = max(0, int((zoom_region[0] - genomic_start) * data_len / genomic_span))
                    end_idx = min(data_len, int((zoom_region[1] - genomic_start) * data_len / genomic_span))

                    if end_idx > start_idx:
                        x = np.linspace(zoom_region[0], zoom_region[1], end_idx - start_idx)
                        data = data[start_idx:end_idx]
                    else:
                        print(f"Warning: Invalid zoom region {zoom_region}")
                        x = np.arange(len(data))
                else:
                    # Direct array indices
                    x = np.arange(zoom_region[0], min(zoom_region[1], len(data)))
                    data = data[zoom_region[0]:min(zoom_region[1], len(data))]
            else:
                x = np.arange(len(data))

            # Apply smoothing
            if smooth:
                from scipy.ndimage import uniform_filter1d
                data = uniform_filter1d(data, smooth)

            # Normalize if requested
            if normalize and data.max() > 0:
                data = data / data.max()

            # Determine color
            color_key = name.lower().replace(' ', '_').replace('(+)', 'plus').replace('(-)', 'minus')
            color = colors.get(color_key, 'gray')

            # Plot
            ax.fill_between(x, data, alpha=0.7, color=color)
            ax.set_title(name)

            # Format x-axis with genomic coordinates
            from matplotlib.ticker import FuncFormatter
            def format_genomic_pos(x, p):
                # x is in data indices, convert to genomic position
                data_len = len(data) if not isinstance(data, dict) else len(data.get('plus', []))
                if data_len > 0:
                    genomic_pos = self.start + (x / data_len) * (self.end - self.start)
                    mb_value = genomic_pos / 1e6
                    return f'{mb_value:.2f}'
                return str(x)

            ax.xaxis.set_major_formatter(FuncFormatter(format_genomic_pos))
            ax.set_xlabel(f'Position on {self.chrom} (Mb)')
            ax.set_ylabel('Signal')

            # Peak calling
            if peaks and data.max() > 0:
                peak_idx, _ = find_peaks(data, height=data.max() * 0.3, distance=100)
                # Limit number of peaks shown to avoid overflow
                if len(peak_idx) > 100:
                    peak_idx = peak_idx[::len(peak_idx)//100]  # Sample to 100 peaks
                if len(peak_idx) > 0 and len(peak_idx) < 1000:
                    ax.scatter(x[peak_idx], data[peak_idx],
                             color='red', s=10, zorder=10, marker='v')

            # Gene highlight
            if highlight_gene and self.gene_start and self.gene_end:
                ax.axvspan(self.gene_start, self.gene_end, alpha=0.2, color='yellow')

        # Hide unused subplots
        for i in range(n_tracks, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(title or f'Regulatory Analysis: {self.chrom}:{self.start:,}-{self.end:,}')
        plt.tight_layout()

        # Save
        if output:
            plt.savefig(output, dpi=dpi, bbox_inches='tight')
            print(f"Saved to {output}")
        else:
            plt.show()


def check_ism_data(npz_path):
    """Check if NPZ contains ISM data and suggest appropriate visualizer"""
    import numpy as np
    data = np.load(npz_path)

    # Check for ISM pattern in keys
    if any('mutation_' in key for key in data.keys()):
        print("\n✅ ISM data detected! Use --tracks ism for heatmap visualization")
        print("Example: python visualizer.py {} --tracks ism -o ism_heatmap.png\n".format(npz_path))
        return True
    return False

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Advanced visualization for AlphaGenome NPZ data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all available tracks with overlays
  python visualizer.py data.npz --tracks all --overlays tss gene

  # View RNA-seq both strands (default behavior)
  python visualizer.py data.npz --tracks rna-seq

  # View specific strand only
  python visualizer.py data.npz --tracks rna-seq:+ rna-seq:-

  # Multiple tracks on single canvas with TSS overlay
  python visualizer.py data.npz --tracks dnase rna-seq histone:h3k27ac --canvas single --overlays tss

  # Each track on separate subplot
  python visualizer.py data.npz --tracks dnase rna-seq chip-tf --canvas multi

  # Specific histone marks in grid
  python visualizer.py data.npz --tracks histone:h3k4me3,h3k27ac,h3k27me3 --canvas grid

  # Publication-ready with smoothing
  python visualizer.py data.npz --tracks dnase rna-seq --style publication --smooth 100 --dpi 300

  # Zoom to specific region with peak calling
  python visualizer.py data.npz --tracks dnase --zoom 250000 270000 --peaks

Track specifications:
  all                - All available tracks
  dnase              - DNase/ATAC accessibility
  rna-seq            - RNA-seq both strands (default)
  rna-seq:+          - RNA-seq plus strand only
  rna-seq:-          - RNA-seq minus strand only
  rna-seq:+-         - RNA-seq both strands (same as rna-seq)
  chip-tf            - Transcription factor ChIP
  histone:all        - All histone marks
  histone:h3k4me3    - Specific histone mark
  cage               - CAGE data
  procap             - ProCap data

Overlay options (use --overlays):
  tss                - Transcription start site line
  gene               - Gene body region
  promoter           - Core and proximal promoter regions
  enhancer           - Upstream enhancer region
        """
    )

    parser.add_argument('npz_file', help='Path to AlphaGenome NPZ file')
    parser.add_argument('--json', help='Path to metadata JSON (auto-detected if not specified)')

    # Track selection
    parser.add_argument('--tracks', nargs='+', required=True,
                       help='Tracks to visualize (see examples)')

    # Canvas options
    parser.add_argument('--canvas', choices=['single', 'multi', 'grid'],
                       default='multi',
                       help='Canvas layout: single (stacked), multi (subplots), grid')
    parser.add_argument('--grid-cols', type=int, default=2,
                       help='Number of columns for grid layout')

    # Styling
    parser.add_argument('--style', choices=['default', 'publication', 'grayscale'],
                       default='default',
                       help='Color scheme')
    parser.add_argument('--figsize', nargs=2, type=float,
                       help='Figure size as width height (e.g., 15 10)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output image')

    # Data processing
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize each track to [0, 1]')
    parser.add_argument('--smooth', type=int,
                       help='Smoothing window size')
    parser.add_argument('--peaks', action='store_true',
                       help='Show peak calls')
    parser.add_argument('--zoom', type=str,
                       help='Zoom format: position:factor (e.g., tss:10, gene:5, promoter:20)')
    parser.add_argument('--window', type=str,
                       help='Window format: start:end genomic coordinates (e.g., 44900000:44950000)')

    # Annotations
    parser.add_argument('--no-gene', action='store_true',
                       help='Do not highlight gene region')
    parser.add_argument('--title', help='Custom title')
    parser.add_argument('--overlays', nargs='+',
                       choices=['tss', 'gene', 'promoter', 'enhancer'],
                       help='Overlay regions to display on tracks (default: tss)')
    parser.add_argument('--no-overlays', action='store_true',
                       help='Disable all overlays including default TSS')
    parser.add_argument('--tss', type=int, help='TSS position for regulatory regions (legacy)')
    parser.add_argument('--gene-start', type=int, help='Gene start position (legacy)')
    parser.add_argument('--gene-end', type=int, help='Gene end position (legacy)')
    parser.add_argument('--regulatory', action='store_true',
                       help='Show regulatory regions (legacy, use --overlays instead)')
    parser.add_argument('--targets', nargs='+', type=int,
                       help='CRISPR target positions to highlight')

    # Output
    parser.add_argument('--output', '-o', help='Output filename')

    args = parser.parse_args()

    # Check file exists
    if not Path(args.npz_file).exists():
        print(f"Error: {args.npz_file} not found")
        sys.exit(1)

    # Check for ISM data and suggest appropriate visualizer
    check_ism_data(args.npz_file)

    # Initialize visualizer
    viz = AlphaGenomeVisualizer(args.npz_file, args.json)

    # Prepare arguments
    # Default to TSS overlay if no overlays specified (unless explicitly disabled)
    # This helps with coordinate scaling
    if args.no_overlays:
        default_overlays = None
    elif args.overlays is None:
        default_overlays = ['tss']
    else:
        default_overlays = args.overlays

    viz_kwargs = {
        'canvas': args.canvas,
        'style': args.style,
        'dpi': args.dpi,
        'output': args.output,
        'highlight_gene': not args.no_gene,
        'normalize': args.normalize,
        'smooth': args.smooth,
        'peaks': args.peaks,
        'title': args.title,
        'grid_cols': args.grid_cols,
        'overlays': default_overlays
    }

    if args.figsize:
        viz_kwargs['figsize'] = tuple(args.figsize)

    # Handle --window parameter: start:end genomic coordinates
    if args.window:
        try:
            start_str, end_str = args.window.split(':')
            window_start = int(start_str)
            window_end = int(end_str)
            viz_kwargs['zoom_region'] = (window_start, window_end)
            print(f"Setting window to {window_start:,}-{window_end:,}")
        except ValueError:
            print(f"Error: Invalid window format '{args.window}'. Use start:end (e.g., 44900000:44950000)")
            sys.exit(1)

    # Handle --zoom parameter: position:factor
    if args.zoom:
        try:
            position, factor_str = args.zoom.split(':')
            zoom_factor = float(factor_str)

            # Determine the center position based on the overlay name
            center_pos = None
            if position.lower() == 'tss' and viz.tss:
                center_pos = viz.tss
                pos_name = "TSS"
            elif position.lower() == 'gene' and viz.gene_start and viz.gene_end:
                center_pos = (viz.gene_start + viz.gene_end) // 2
                pos_name = "gene center"
            elif position.lower() == 'promoter' and viz.tss:
                # Center on core promoter region
                center_pos = viz.tss
                pos_name = "promoter (TSS)"
            elif position.lower() == 'enhancer' and viz.tss:
                # Center on enhancer region (roughly -3.5kb from TSS)
                center_pos = viz.tss - 3500
                pos_name = "enhancer region"
            else:
                print(f"Error: Cannot find position '{position}' or it's not available in metadata")
                sys.exit(1)

            if center_pos:
                # Calculate zoom window
                current_window = viz.end - viz.start
                new_window = int(current_window / zoom_factor)
                half_window = new_window // 2

                zoom_start = max(viz.start, int(center_pos - half_window))
                zoom_end = min(viz.end, int(center_pos + half_window))

                viz_kwargs['zoom_region'] = (zoom_start, zoom_end)
                print(f"Zooming {zoom_factor}x on {pos_name} at {center_pos:,} (window: {new_window:,}bp)")
        except ValueError:
            print(f"Error: Invalid zoom format '{args.zoom}'. Use position:factor (e.g., tss:10)")
            sys.exit(1)

    # Add regulatory regions if requested
    if args.regulatory and args.tss:
        viz_kwargs['tss'] = args.tss
        viz_kwargs['show_regulatory'] = True
        if args.gene_start:
            viz_kwargs['gene_start'] = args.gene_start
        if args.gene_end:
            viz_kwargs['gene_end'] = args.gene_end

    if args.targets:
        viz_kwargs['crispr_targets'] = args.targets

    # Visualize
    viz.visualize(args.tracks, **viz_kwargs)


if __name__ == '__main__':
    main()