#!/usr/bin/env python3
import sys
import warnings
warnings.filterwarnings('ignore')

import celloracle as co
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Global variable for GRN
CURRENT_GRN = None
GRN_SOURCE = 'promoter'

def load_merged_grn():
    """Load the pre-merged TRRUST+Promoter GRN"""
    global CURRENT_GRN, GRN_SOURCE

    merged_path = Path('merged_grn_promoter_trrust.pkl')
    if not merged_path.exists():
        print('❌ Merged GRN not found. Run merge_trrust_grn.py first.')
        return False

    print('Loading merged TRRUST+Promoter GRN...')
    with open(merged_path, 'rb') as f:
        CURRENT_GRN = pickle.load(f)

    GRN_SOURCE = 'merged'
    print(f'✓ Loaded merged GRN: {len(CURRENT_GRN)} genes')

    # Count unique TFs
    all_tfs = set()
    for tfs in CURRENT_GRN.values():
        for tf in tfs:
            if isinstance(tf, str):
                all_tfs.add(tf)
    print(f'  {len(all_tfs)} unique TFs')

    # Check improvements
    improvements = []
    if 'GFAP' in CURRENT_GRN and 'STAT3' in CURRENT_GRN['GFAP']:
        improvements.append('GFAP→STAT3')
    if 'IL1B' in CURRENT_GRN and 'NFKB1' in CURRENT_GRN['IL1B']:
        improvements.append('IL1B→NFKB1')

    if improvements:
        print(f'  ✓ Key improvements: {", ".join(improvements)}')

    return True

def load_custom_grn(grn_file, merge=False):
    """Load custom GRN from CSV file"""
    global CURRENT_GRN, GRN_SOURCE

    print(f'Loading custom GRN from {grn_file}...')

    grn_path = Path(grn_file)
    if not grn_path.exists():
        print(f'❌ GRN file not found: {grn_file}')
        return False

    # Load custom GRN
    custom_grn = pd.read_csv(grn_path)

    # Check required columns
    required = ['gene_short_name', 'TF']
    if not all(col in custom_grn.columns for col in required):
        # Also check alternative column names
        if 'gene' in custom_grn.columns:
            custom_grn['gene_short_name'] = custom_grn['gene']
        else:
            print(f'❌ GRN must have columns: {required}')
            return False

    print(f'✓ Loaded {len(custom_grn)} TF-gene interactions')
    print(f'  {custom_grn["TF"].nunique()} unique TFs')
    print(f'  {custom_grn["gene_short_name"].nunique()} unique genes')

    # Convert to dictionary format
    custom_dict = {}
    for gene in custom_grn['gene_short_name'].unique():
        tfs = custom_grn[custom_grn['gene_short_name'] == gene]['TF'].unique().tolist()
        custom_dict[gene] = tfs

    if merge and CURRENT_GRN is not None:
        # Merge with existing GRN
        print('Merging with existing GRN...')
        all_genes = set(CURRENT_GRN.keys()) | set(custom_dict.keys())
        merged = {}

        for gene in all_genes:
            existing_tfs = set(CURRENT_GRN.get(gene, []))
            custom_tfs = set(custom_dict.get(gene, []))
            all_tfs = list(existing_tfs | custom_tfs)
            if all_tfs:
                merged[gene] = all_tfs

        print(f'✓ Merged GRN: {len(merged)} genes')
        CURRENT_GRN = merged
        GRN_SOURCE = 'merged_custom'
    else:
        # Replace with custom GRN
        CURRENT_GRN = custom_dict
        GRN_SOURCE = 'custom'
        print(f'✓ Using custom GRN: {len(custom_dict)} genes')

    return True

def find_controllers(gene_name):
    """Find TF controllers for a gene"""
    global CURRENT_GRN, GRN_SOURCE

    print(f'\nFinding TF controllers for {gene_name}')
    print('='*60)

    gene_name = gene_name.upper()

    # First check if using custom/merged GRN
    if CURRENT_GRN is not None:
        print(f'Using {GRN_SOURCE} GRN...')

        if gene_name in CURRENT_GRN:
            tfs = CURRENT_GRN[gene_name]
            # Filter out non-string values
            tfs = [tf for tf in tfs if isinstance(tf, str)]

            print(f'\nFound {len(tfs)} TFs regulating {gene_name}:')
            print('-'*60)

            # Sort alphabetically
            tfs_sorted = sorted(tfs)

            # Display in columns
            for i in range(0, len(tfs_sorted), 3):
                row_tfs = tfs_sorted[i:i+3]
                line = '  '
                for tf in row_tfs:
                    line += f'{tf:25s}'
                print(line)

            # Check for key astrocyte TFs
            print(f'\nKEY TFS PRESENT:')
            print('-'*60)

            astro_tfs = ['STAT3', 'SOX9', 'NFIA', 'NFIB', 'NFIC']
            present = [tf for tf in astro_tfs if tf in tfs]
            if present:
                print(f'Astrocyte TFs: {", ".join(present)}')

            nfkb = [tf for tf in tfs if tf in ['NFKB1', 'RELA', 'REL', 'RELB']]
            if nfkb:
                print(f'NF-κB family: {", ".join(nfkb)}')

            stat = [tf for tf in tfs if tf.startswith('STAT')]
            if stat:
                print(f'STAT family: {", ".join(stat)}')

            return tfs
        else:
            print(f'{gene_name} not found in {GRN_SOURCE} GRN')
            return []

    # Fall back to promoter GRN
    print('Loading promoter GRN...')
    base_grn = co.data.load_human_promoter_base_GRN('hg38_gimmemotifsv5_fpr1')

    # Check if gene exists
    if gene_name not in base_grn['gene_short_name'].values:
        print(f'{gene_name} not found in promoter GRN')
        return []

    # Get TF columns
    tf_columns = [col for col in base_grn.columns if col not in ['peak_id', 'gene_short_name']]

    # Get gene rows
    gene_rows = base_grn[base_grn['gene_short_name'] == gene_name]
    print(f'Found {len(gene_rows)} regulatory region(s) for {gene_name}')

    # Calculate TF scores
    tf_scores = {}
    for tf in tf_columns:
        score = gene_rows[tf].sum()
        if score > 0:
            tf_scores[tf] = score

    if not tf_scores:
        print(f'No TF binding sites found for {gene_name}')
        return []

    # Show results
    print(f'\nTotal TFs with binding sites: {len(tf_scores)}')
    print('-'*60)

    # Sort by score
    sorted_tfs = sorted(tf_scores.items(), key=lambda x: (-x[1], x[0]))

    for i, (tf, score) in enumerate(sorted_tfs[:50]):
        print(f'  {tf:20s} Score: {score:.1f}')
        if i == 49 and len(sorted_tfs) > 50:
            print(f'  ... and {len(sorted_tfs)-50} more')

    return list(tf_scores.keys())

def find_targets(tf_name):
    """Find targets of a TF"""
    global CURRENT_GRN, GRN_SOURCE

    print(f'\nFinding targets of {tf_name}')
    print('='*60)

    tf_name = tf_name.upper()

    # Check if using custom/merged GRN
    if CURRENT_GRN is not None:
        print(f'Using {GRN_SOURCE} GRN...')

        targets = []
        for gene, tfs in CURRENT_GRN.items():
            if tf_name in tfs:
                targets.append(gene)

        if targets:
            print(f'\n{tf_name} regulates {len(targets)} genes:')
            print('-'*60)

            # Sort alphabetically
            targets_sorted = sorted(targets)

            # Show first 100
            for i, gene in enumerate(targets_sorted[:100]):
                if i % 5 == 0:
                    print()
                    print('  ', end='')
                print(f'{gene:15s}', end='')

            if len(targets) > 100:
                print(f'\n  ... and {len(targets)-100} more')

            print()
            return targets
        else:
            print(f'{tf_name} has no targets in {GRN_SOURCE} GRN')
            return []

    # Fall back to promoter GRN
    print('Loading promoter GRN...')
    base_grn = co.data.load_human_promoter_base_GRN('hg38_gimmemotifsv5_fpr1')

    if tf_name not in base_grn.columns:
        print(f'{tf_name} not found as TF in promoter GRN')
        return []

    # Find all targets
    tf_targets = base_grn[base_grn[tf_name] > 0]

    if len(tf_targets) == 0:
        print(f'No targets found for {tf_name}')
        return []

    # Group by gene
    gene_scores = tf_targets.groupby('gene_short_name')[tf_name].sum()

    print(f'{tf_name} has binding sites at {len(gene_scores)} genes')

    # Show top targets
    top_targets = gene_scores.sort_values(ascending=False).head(50)
    print(f'\nTop 50 targets (of {len(gene_scores)} total):')
    print('-'*50)
    for gene, score in top_targets.items():
        print(f'  {gene:20s} Score: {score:.0f}')

    return list(gene_scores.index)

def validate_known():
    """Validate known TF-gene relationships"""
    global CURRENT_GRN, GRN_SOURCE

    print('\nValidating Known TF-Gene Relationships')
    print('='*60)
    print(f'Using {GRN_SOURCE} GRN')
    print()

    # Test cases
    test_cases = [
        ('STAT3', ['GFAP', 'AQP4', 'S100B', 'IL6', 'BCL2']),
        ('SOX9', ['COL2A1', 'COL11A2', 'ACAN']),
        ('NFKB1', ['IL1B', 'IL6', 'TNF', 'ICAM1']),
        ('NFIA', ['GFAP', 'S100B']),
    ]

    for tf, genes in test_cases:
        print(f'{tf} → expected targets:')
        for gene in genes:
            if CURRENT_GRN and gene in CURRENT_GRN:
                if tf in CURRENT_GRN[gene]:
                    print(f'  ✓ {gene}')
                else:
                    print(f'  ✗ {gene}')
            else:
                print(f'  ? {gene} (not in GRN)')
        print()

def perturb_gene(gene_name, knockdown_level=0.0):
    """Simulate gene perturbation"""
    global CURRENT_GRN, GRN_SOURCE
    
    print(f'\nSimulating Perturbation: {gene_name}')
    print('='*60)
    
    gene_name = gene_name.upper()
    
    # Determine perturbation type
    if knockdown_level == 0.0:
        perturb_type = "Complete Knockdown"
    elif knockdown_level < 0.5:
        perturb_type = f"Strong Knockdown ({100*(1-knockdown_level):.0f}% reduction)"
    elif knockdown_level < 1.0:
        perturb_type = f"Partial Knockdown ({100*(1-knockdown_level):.0f}% reduction)"  
    elif knockdown_level == 1.0:
        perturb_type = "No Change (Control)"
    else:
        perturb_type = f"Overexpression ({knockdown_level:.1f}x)"
    
    print(f'Gene: {gene_name}')
    print(f'Expression Level: {knockdown_level:.2f}')
    print(f'Perturbation Type: {perturb_type}')
    print()
    
    # Load GRN if not already loaded
    if CURRENT_GRN is None:
        print('Loading default GRN...')
        load_merged_grn()
    
    # Find direct targets if gene is a TF
    targets = find_targets(gene_name)
    if targets:
        print(f'\n{gene_name} is a TF with {len(targets)} direct targets')
        print('-'*60)
        
        # Show top affected targets
        print('\nTop 10 directly affected genes:')
        for i, target in enumerate(targets[:10]):
            if knockdown_level == 0.0:
                effect = "Lost regulation"
            elif knockdown_level < 1.0:
                effect = f"Reduced regulation ({knockdown_level:.1f}x)"
            else:
                effect = f"Enhanced regulation ({knockdown_level:.1f}x)"
            print(f'  {i+1:2}. {target:15s} - {effect}')
    
    # Find upstream regulators
    regulators = find_controllers(gene_name)
    if regulators:
        print(f'\n{gene_name} is regulated by {len(regulators)} TFs')
        print('-'*60)
        print('\nTop upstream regulators (may compensate):')
        for i, tf in enumerate(regulators[:10]):
            print(f'  {i+1:2}. {tf}')
    
    # Network impact analysis
    print('\n' + '='*60)
    print('NETWORK IMPACT ANALYSIS')
    print('='*60)
    
    if targets:
        # Count downstream cascade
        cascade_genes = set(targets)
        second_order = set()
        
        for target_gene in targets[:50]:  # Check first 50 for efficiency
            target_targets = find_targets(target_gene)
            second_order.update(target_targets[:20])  # Add first 20 second-order targets
        
        total_affected = len(cascade_genes) + len(second_order)
        
        print(f'\nNetwork Cascade Effects:')
        print(f'  Direct targets: {len(targets)}')
        print(f'  Secondary targets (estimated): {len(second_order)}')
        print(f'  Total affected genes: ~{total_affected}')
        
        # Pathway analysis
        print(f'\nPathway Impact:')
        
        # Check for inflammatory genes
        inflammatory = ['IL1B', 'IL6', 'TNF', 'CXCL10', 'CCL2', 'NFKB1', 'RELA']
        affected_inflammatory = [g for g in inflammatory if g in cascade_genes]
        if affected_inflammatory:
            print(f'  Inflammatory: {", ".join(affected_inflammatory)}')
        
        # Check for astrocyte genes
        astrocyte = ['GFAP', 'S100B', 'AQP4', 'SLC1A2', 'SLC1A3', 'ALDH1L1']
        affected_astro = [g for g in astrocyte if g in cascade_genes]
        if affected_astro:
            print(f'  Astrocyte markers: {", ".join(affected_astro)}')
        
        # Check for memory genes (subset)
        memory = ['ARC', 'FOS', 'JUN', 'EGR1', 'BDNF', 'CREB1']
        affected_memory = [g for g in memory if g in cascade_genes]
        if affected_memory:
            print(f'  Memory-related: {", ".join(affected_memory)}')
    
    # Suggest validation
    print('\n' + '='*60)
    print('EXPERIMENTAL VALIDATION')
    print('='*60)
    
    print('\nSuggested validation experiments:')
    print('  1. qPCR validation of top 5 predicted targets')
    print('  2. Western blot for protein level changes')
    print('  3. RNA-seq for genome-wide validation')
    
    if knockdown_level == 0.0:
        print('\nCRISPR Design:')
        print(f'  Target gene: {gene_name}')
        print('  Suggested approach: CRISPRi for knockdown')
        print('  Use AlphaGenome to identify optimal guide positions')
    
    return {
        'gene': gene_name,
        'knockdown': knockdown_level,
        'direct_targets': len(targets) if targets else 0,
        'upstream_regulators': len(regulators) if regulators else 0,
        'network_impact': total_affected if targets else 0
    }

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='CellOracle CLI - Enhanced GRN Analysis')
    p.add_argument('--use-merged-grn', action='store_true',
                   help='Use pre-merged TRRUST+Promoter GRN')
    p.add_argument('--load-custom-grn', help='Load custom GRN from CSV file')
    p.add_argument('--merge', action='store_true',
                   help='Merge custom GRN with existing (use with --load-custom-grn)')
    p.add_argument('--find-regulators', help='Find TFs regulating a gene')
    p.add_argument('--find-targets', help='Find genes regulated by a TF')
    p.add_argument('--validate', action='store_true', help='Validate known relationships')
    
    # New perturbation arguments
    p.add_argument('--perturb', help='Gene to perturb')
    p.add_argument('--knockdown', type=float, default=0.0,
                   help='Expression level after perturbation (0.0=complete KD, 1.0=no change, >1.0=overexpression)')

    args = p.parse_args()

    # Load GRN if specified
    if args.use_merged_grn:
        if not load_merged_grn():
            sys.exit(1)

    if args.load_custom_grn:
        if not load_custom_grn(args.load_custom_grn, merge=args.merge):
            sys.exit(1)

    # Execute commands
    if args.perturb:
        # Handle perturbation
        perturb_gene(args.perturb, args.knockdown)
    elif args.find_regulators:
        find_controllers(args.find_regulators.upper())
    elif args.find_targets:
        find_targets(args.find_targets.upper())
    elif args.validate:
        validate_known()
    else:
        print('CellOracle CLI - Enhanced GRN Analysis')
        print()
        print('GRN Loading Options:')
        print('  --use-merged-grn        : Use TRRUST+Promoter merged GRN')
        print('  --load-custom-grn FILE  : Load custom GRN from CSV')
        print('  --merge                 : Merge custom with existing')
        print()
        print('Analysis Commands:')
        print('  --find-regulators GENE  : Find TFs regulating a gene')
        print('  --find-targets TF       : Find genes regulated by a TF')
        print('  --validate              : Validate known relationships')
        print()
        print('Perturbation Commands:')
        print('  --perturb GENE          : Simulate gene perturbation')
        print('  --knockdown LEVEL       : Expression level (0.0=KD, 1.0=normal, >1.0=OE)')
        print()
        print('Examples:')
        print('  python CellOracle.py --use-merged-grn --find-regulators SOX9')
        print('  python CellOracle.py --use-merged-grn --find-regulators GFAP')
        print('  python CellOracle.py --use-merged-grn --find-targets STAT3')
        print('  python CellOracle.py --load-custom-grn my_grn.csv --find-regulators IL1B')
        print('  python CellOracle.py --perturb STAT3 --knockdown 0.0')
        print('  python CellOracle.py --perturb JUN --knockdown 0.5')
