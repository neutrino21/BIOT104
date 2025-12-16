#!/usr/bin/env python3
"""
Integrated CRISPR Target Scoring System
Combines ISM impact + TF motifs + multi-gene effects

This solves: "Which CRISPR target gives maximum therapeutic impact?"
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path


class IntegratedCRISPRScoring:
    def __init__(self, targets_file, activate_genes=None, binary_mode=None, binary_genes=None):
        """
        Load CRISPR targets from TSV
        Format: gene, mode, chromosome, position, impact, distance_to_tss [, motif_type]

        Args:
            targets_file: Path to TSV with CRISPR targets
            activate_genes: List of genes to activate (rest will be inhibited) - INTEGRATED MODE
            binary_mode: 'activate' or 'inhibit' - BINARY MODE
            binary_genes: List of genes for binary mode - BINARY MODE
        """
        self.activate_genes = set(activate_genes) if activate_genes else set()
        self.binary_mode = binary_mode
        self.binary_genes = set(binary_genes) if binary_genes else set()
        # Read TSV file properly
        self.df = pd.read_csv(targets_file, sep='\t')

        # Check if this is the new format with headers
        if 'impact' not in self.df.columns and 'ism_max' in self.df.columns:
            self.df['impact'] = self.df['ism_max']

        # Ensure required columns exist
        required_cols = ['gene', 'mode', 'chromosome', 'position', 'impact', 'distance_to_tss']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            print(f"Available columns: {self.df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add motif_type column if it doesn't exist
        if 'motif_type' not in self.df.columns:
            self.df['motif_type'] = None

        # Convert numeric columns
        self.df['position'] = pd.to_numeric(self.df['position'], errors='coerce')
        self.df['impact'] = pd.to_numeric(self.df['impact'], errors='coerce')
        self.df['distance_to_tss'] = pd.to_numeric(self.df['distance_to_tss'], errors='coerce')

        # Remove any rows with NaN in critical columns
        self.df = self.df.dropna(subset=['position', 'impact'])

        # Filter out motif-only rows for main analysis
        self.df_targets = self.df[~self.df['mode'].str.startswith('motif_')]

        # Known TF motifs and their therapeutic importance
        self.tf_motifs = {
            'GGAA': {'name': 'NF-κB', 'weight': 2.0, 'therapeutic': 'Master inflammatory regulator'},
            'NF-κB': {'name': 'NF-κB', 'weight': 2.0, 'therapeutic': 'Master inflammatory regulator'},
            'NF-κB_ext': {'name': 'NF-κB', 'weight': 2.0, 'therapeutic': 'Master inflammatory regulator'},
            'NF-κB_var': {'name': 'NF-κB', 'weight': 2.0, 'therapeutic': 'Master inflammatory regulator'},
            'TGACTCA': {'name': 'AP-1', 'weight': 1.8, 'therapeutic': 'Stress response'},
            'AP-1': {'name': 'AP-1', 'weight': 1.8, 'therapeutic': 'Stress response'},
            'AP-1_var': {'name': 'AP-1', 'weight': 1.8, 'therapeutic': 'Stress response'},
            'GCAAT': {'name': 'C/EBP', 'weight': 1.5, 'therapeutic': 'Acute phase response'},
            'C/EBP': {'name': 'C/EBP', 'weight': 1.5, 'therapeutic': 'Acute phase response'},
            'TTCNNGAA': {'name': 'STAT', 'weight': 1.7, 'therapeutic': 'Cytokine signaling'},
            'STAT': {'name': 'STAT', 'weight': 1.7, 'therapeutic': 'Cytokine signaling'},
            'CAAT': {'name': 'CAAT box', 'weight': 1.3, 'therapeutic': 'General transcription'},
            'CAAT_box': {'name': 'CAAT box', 'weight': 1.3, 'therapeutic': 'General transcription'},
            'TATAAA': {'name': 'TATA box', 'weight': 1.2, 'therapeutic': 'General transcription'},
            'TATA_box': {'name': 'TATA box', 'weight': 1.2, 'therapeutic': 'General transcription'},
            'GGGCGG': {'name': 'SP1', 'weight': 1.2, 'therapeutic': 'Basal transcription'},
            'SP1': {'name': 'SP1', 'weight': 1.2, 'therapeutic': 'Basal transcription'}
        }

        self.scored_targets = []

    def get_sequence(self, chromosome, position, window=30):
        """Get DNA sequence around position"""
        chrom = chromosome.replace('chr', '')
        server = "https://rest.ensembl.org"
        ext = f"/sequence/region/human/{chrom}:{position-window}..{position+window}:1?"

        try:
            r = requests.get(server + ext, headers={"Content-Type": "text/plain"})
            if r.ok:
                return r.text.strip().upper()
        except:
            pass
        return ""

    def find_motifs_in_sequence(self, sequence):
        """Find all TF motifs in a sequence"""
        found_motifs = []

        for motif, info in self.tf_motifs.items():
            # Handle degenerate bases
            search_pattern = motif.replace('N', '.')
            if search_pattern in sequence or \
               any(search_pattern.replace('.', base) in sequence for base in 'ACGT'):
                found_motifs.append((motif, info))

        return found_motifs

    def calculate_integrated_score(self):
        """
        Calculate integrated score for each target:
        SCORE = ISM_impact × motif_multiplier × gene_direction_factor

        gene_direction_factor:
        - All genes same direction needed: sqrt(gene_count)
        - Mixed directions (conflict): -1 (negative score, avoid this position)
        """

        print("="*80)
        print("INTEGRATED CRISPR TARGET SCORING")
        print("="*80)

        if self.binary_mode:
            # BINARY MODE: Only consider specific genes for one action
            print(f"\n🎯 BINARY MODE: {self.binary_mode.upper()}")
            print(f"Target genes: {', '.join(sorted(self.binary_genes))}")
            print("\nOnly positions affecting these genes will be scored.")
            print("Scoring formula: ISM_impact × motif_weight × sqrt(gene_count)\n")
        elif self.activate_genes:
            # INTEGRATED MODE: Mixed activation/inhibition
            print(f"\n🔬 INTEGRATED MODE")
            print(f"Activation targets: {', '.join(sorted(self.activate_genes))}")
            inhibit_genes = set(self.df_targets['gene'].unique()) - self.activate_genes
            print(f"Inhibition targets: {', '.join(sorted(inhibit_genes))}")
            print("\nScoring formula: ISM_impact × motif_weight × gene_direction_factor")
            print("where gene_direction_factor = sqrt(gene_count) if consistent, -1 if conflict\n")
        else:
            # DEFAULT: All genes for inhibition
            print("\n📊 DEFAULT MODE: All genes for inhibition")
            print("Scoring formula: ISM_impact × motif_weight × sqrt(gene_count)\n")

        # First, score regular ISM targets
        position_groups = self.df_targets.groupby(['chromosome', 'position'])

        print(f"DEBUG: Found {len(position_groups)} position groups")
        print(f"DEBUG: df_targets shape: {self.df_targets.shape}")

        for (chrom, pos), group in position_groups:
            # Get unique genes at this position
            genes = group['gene'].unique()
            gene_count = len(genes)

            # BINARY MODE: Skip if genes not in target list
            if self.binary_mode:
                genes_in_target = [g for g in genes if g in self.binary_genes]
                if not genes_in_target:
                    print(f"  Skipping {chrom}:{pos} - no target genes")
                    continue  # Skip this position
                # Only consider target genes
                genes = genes_in_target
                gene_count = len(genes)
                # Filter group to only target genes
                group = group[group['gene'].isin(self.binary_genes)]
                # Also filter by mode
                if self.binary_mode == 'activate':
                    group = group[group['mode'].str.contains('activat')]
                elif self.binary_mode == 'inhibit':
                    group = group[group['mode'].str.contains('inhibit')]

                if group.empty:
                    print(f"  Skipping {chrom}:{pos} - no matching mode data")
                    continue  # Skip if no matching mode data

            # Check for direction conflict (only in integrated mode)
            genes_to_activate = set(g for g in genes if g in self.activate_genes)
            genes_to_inhibit = set(g for g in genes if g not in self.activate_genes)

            # INTEGRATED/DEFAULT MODE: Filter by appropriate mode
            if not self.binary_mode and (genes_to_activate or genes_to_inhibit):
                # Filter group to only include appropriate mode for each gene
                activate_mask = (group['gene'].isin(genes_to_activate) &
                                group['mode'].str.contains('activat'))
                inhibit_mask = (group['gene'].isin(genes_to_inhibit) &
                               group['mode'].str.contains('inhibit'))

                # For default mode (all inhibition), just use inhibition
                if not self.activate_genes:
                    group = group[group['mode'].str.contains('inhibit')]
                else:
                    group = group[activate_mask | inhibit_mask]

                if group.empty:
                    continue  # Skip if no appropriate mode data

                # Recount genes after filtering
                genes = group['gene'].unique()
                gene_count = len(genes)

            print(f"\n📍 Analyzing {chrom}:{pos}")
            avg_impact = group['impact'].mean()

            # Determine if we need activation or inhibition at this position
            # Look at the mode column to see what this position does
            modes_at_position = group['mode'].unique()
            has_activation = any('activat' in m for m in modes_at_position)
            has_inhibition = any('inhibit' in m for m in modes_at_position)

            direction_conflict = False
            position_strategy = "inhibition"  # default

            if self.binary_mode:
                # BINARY MODE: Strategy is determined by the action
                position_strategy = self.binary_mode
            elif self.activate_genes:
                # INTEGRATED MODE: Determine strategy based on genes at this position
                if genes_to_activate and genes_to_inhibit:
                    # This position affects both types - conflict!
                    direction_conflict = True
                    print(f"  ⚠️ CONFLICT: Position affects both activation ({', '.join(genes_to_activate)}) "
                          f"and inhibition ({', '.join(genes_to_inhibit)}) targets")
                elif genes_to_activate:
                    # Only affects genes we want to activate
                    position_strategy = "activation"
                else:
                    # Only affects genes we want to inhibit
                    position_strategy = "inhibition"

            print(f"  Genes affected: {', '.join(genes)} (n={gene_count})")
            print(f"  Strategy needed: {position_strategy}")
            print(f"  Average ISM impact: {avg_impact:.1f}")

            # Get DNA sequence
            sequence = self.get_sequence(chrom, pos)

            # Check if we have motif data from the file
            motif_rows = self.df[(self.df['mode'].str.startswith('motif_')) &
                                (self.df['chromosome'] == chrom) &
                                (abs(self.df['position'] - pos) <= 20)]

            # Find motifs
            motifs = []
            if not motif_rows.empty:
                # Use motif data from file
                for _, row in motif_rows.iterrows():
                    if pd.notna(row['motif_type']):
                        motif_info = self.tf_motifs.get(row['motif_type'],
                                                       {'name': row['motif_type'],
                                                        'weight': 1.5,
                                                        'therapeutic': 'TF binding site'})
                        motifs.append((row['motif_type'], motif_info))
            else:
                # Fallback to sequence search
                motifs = self.find_motifs_in_sequence(sequence)

            # Calculate motif multiplier
            motif_weight = 1.0
            motif_names = []

            if motifs:
                # Use highest weight motif
                max_motif = max(motifs, key=lambda x: x[1]['weight'])
                motif_weight = max_motif[1]['weight']
                motif_names = [f"{m[1]['name']} ({m[1]['therapeutic']})" for m in motifs]

                print(f"  TF motifs found: {', '.join(motif_names)}")
                print(f"  Motif weight: {motif_weight:.1f}x")
            else:
                print(f"  No known TF motifs found")

            # Gene direction factor
            if direction_conflict:
                # Conflict - negative score to avoid this position
                gene_factor = -1.0
                integrated_score = -abs(avg_impact * motif_weight)  # Negative score
                print(f"  ❌ CONFLICT PENALTY: Score set to {integrated_score:.1f}")
            else:
                # No conflict - normal scoring
                gene_factor = np.sqrt(gene_count)
                integrated_score = avg_impact * motif_weight * gene_factor
                print(f"  💫 INTEGRATED SCORE: {integrated_score:.1f}")
                print(f"     = {avg_impact:.1f} (ISM) × {motif_weight:.1f} (motif) × {gene_factor:.2f} (genes)")

            # Store result
            self.scored_targets.append({
                'position': f"{chrom}:{pos}",
                'genes': ', '.join(genes),
                'gene_count': gene_count,
                'strategy': 'CONFLICT' if direction_conflict else position_strategy,
                'ism_impact': avg_impact,
                'motifs': ', '.join([m[1]['name'] for m in motifs]) if motifs else 'None',
                'motif_weight': motif_weight,
                'integrated_score': integrated_score,
                'therapeutic_notes': motif_names[0] if motif_names else 'Direct gene regulation'
            })

        # Now score motif positions from the file
        motif_positions = self.df[self.df['mode'].str.startswith('motif_')]

        if not motif_positions.empty:
            print("\n📍 Scoring motif positions from file...")

            for _, row in motif_positions.iterrows():
                if pd.notna(row['motif_type']):
                    # Get motif weight
                    motif_info = self.tf_motifs.get(row['motif_type'],
                                                   {'name': row['motif_type'],
                                                    'weight': 1.5,
                                                    'therapeutic': 'TF binding site'})

                    # Only process if we haven't already scored this position
                    pos_str = f"{row['chromosome']}:{int(row['position'])}"
                    already_scored = any(t['position'] == pos_str for t in self.scored_targets)

                    if not already_scored:
                        # Use the impact from the nearby ISM peak
                        motif_weight = motif_info['weight']
                        gene_bonus = 1.0  # Single gene for now
                        integrated_score = row['impact'] * motif_weight * gene_bonus

                        # Check strategy for this gene
                        strategy = 'activation' if row['gene'] in self.activate_genes else 'inhibition'

                        self.scored_targets.append({
                            'position': pos_str,
                            'genes': row['gene'],
                            'gene_count': 1,
                            'strategy': strategy,
                            'ism_impact': row['impact'],
                            'motifs': row['motif_type'],
                            'motif_weight': motif_weight,
                            'integrated_score': integrated_score,
                            'therapeutic_notes': f"{motif_info['name']} ({motif_info['therapeutic']})"
                        })

                        print(f"    {pos_str}: {row['motif_type']} motif, score={integrated_score:.1f}")

        # Sort by integrated score
        self.scored_targets.sort(key=lambda x: x['integrated_score'], reverse=True)

        return self.scored_targets

    def display_recommendations(self):
        """Show top therapeutic targets with dual approach"""

        print("\n" + "="*80)
        print("DUAL CRISPR TARGETING RECOMMENDATIONS")
        print("="*80)

        # Exclude conflict positions (negative scores)
        valid_targets = [t for t in self.scored_targets if t['integrated_score'] > 0]

        # Find best ISM-only target (precision)
        ism_only = [t for t in valid_targets if t['motif_weight'] == 1.0]
        best_precision = ism_only[0] if ism_only else (valid_targets[0] if valid_targets else None)

        # Find best motif target (network) - highest scoring target with motif weight > 1.0
        motif_targets = [t for t in valid_targets if t['motif_weight'] > 1.0]
        best_network = motif_targets[0] if motif_targets else None

        print("\n🎯 PRECISION TARGET (ISM-based, maximum functional impact):")
        if best_precision:
            print(f"  Position: {best_precision['position']}")
            print(f"  ISM impact: {best_precision['ism_impact']:.1f}")
            print(f"  Genes: {best_precision['genes']}")
            print(f"  Strategy: Direct functional disruption via dCas9-KRAB")
        else:
            print("  No valid ISM targets found")

        if best_network:
            print("\n🌐 NETWORK TARGET (Motif-based, TF binding disruption):")
            print(f"  Position: {best_network['position']}")
            print(f"  Motif: {best_network['motifs']}")
            print(f"  ISM impact: {best_network['ism_impact']:.1f}")
            print(f"  Integrated score: {best_network['integrated_score']:.1f}")
            print(f"  Therapeutic note: {best_network['therapeutic_notes']}")
            print(f"  Strategy: Block TF binding to affect regulatory network")
        else:
            print("\n🌐 NETWORK TARGET: No TF motifs found near high-impact sites")

        print("\n" + "="*80)
        print("TOP TARGETS BY CATEGORY")
        print("="*80)

        # Separate by category
        ism_only = [t for t in self.scored_targets if t['motif_weight'] == 1.0]
        network_targets = [t for t in self.scored_targets if t['motif_weight'] > 1.0]

        print("\nTop 5 ISM-Only Targets:")
        for i, target in enumerate(ism_only[:5], 1):
            print(f"  #{i}. {target['position']} (Score: {target['integrated_score']:.1f}, Gene: {target['genes']})")

        print("\nTop 5 Network Targets (with TF motifs):")
        if network_targets:
            for i, target in enumerate(network_targets[:5], 1):
                print(f"  #{i}. {target['position']} (Score: {target['integrated_score']:.1f}, Motif: {target['motifs']})")
        else:
            print("  No targets with TF motifs found")

        # Save results
        df_results = pd.DataFrame(self.scored_targets)
        df_results.to_csv('integrated_crispr_scores.csv', index=False)

        print(f"\n💾 Full results saved to integrated_crispr_scores.csv")


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Score CRISPR targets with integrated ISM + motif + gene direction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  BINARY:     Score targets for specific genes with one action
              python crispr_score.py targets.tsv --action inhibit --genes IL1B,TNF

  INTEGRATED: Mixed activation/inhibition with conflict detection
              python crispr_score.py targets.tsv --activate-genes HES5,SOX9

  DEFAULT:    All genes for inhibition
              python crispr_score.py targets.tsv
        """
    )
    parser.add_argument('targets_file', help='TSV file with CRISPR targets')

    # Binary mode
    parser.add_argument('--action', choices=['activate', 'inhibit'],
                       help='Binary mode: action to take on specified genes')
    parser.add_argument('--genes',
                       help='Binary mode: comma-separated genes for the action')

    # Integrated mode
    parser.add_argument('--activate-genes',
                       help='Integrated mode: genes to activate (others inhibited)')

    args = parser.parse_args()

    # Validate arguments
    if args.action and not args.genes:
        parser.error("--action requires --genes")
    if args.genes and not args.action:
        parser.error("--genes requires --action")
    if args.action and args.activate_genes:
        parser.error("Cannot use both binary mode (--action) and integrated mode (--activate-genes)")

    # Parse parameters
    activate_genes = None
    binary_mode = None
    binary_genes = None

    if args.action:
        # BINARY MODE
        binary_mode = args.action
        binary_genes = [g.strip().upper() for g in args.genes.split(',')]
    elif args.activate_genes:
        # INTEGRATED MODE
        activate_genes = [g.strip().upper() for g in args.activate_genes.split(',')]

    # Run integrated scoring
    scorer = IntegratedCRISPRScoring(args.targets_file, activate_genes, binary_mode, binary_genes)
    scorer.calculate_integrated_score()
    scorer.display_recommendations()


if __name__ == "__main__":
    main()