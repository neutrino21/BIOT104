#!/usr/bin/env python3
"""
CRISPR Bulk Aggregator - Aggregate cascade analyses across multiple genes
Finds convergent intervention points and master regulators
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
from crispr import CRISPRCascadeAnalyzer


class CRISPRBulkAggregator:
    """
    Aggregate cascade analyses to find master intervention points
    """

    def __init__(self, memory_genes: List[str], organism: str = 'human'):
        self.memory_genes = [g.upper() for g in memory_genes]
        self.organism = organism
        self.cascade_results = {}
        self.master_regulators = {}
        self.convergence_points = {}
        self.strategies = {
            'inhibition': defaultdict(lambda: {'genes': [], 'score': 0, 'mechanisms': set()}),
            'activation': defaultdict(lambda: {'genes': [], 'score': 0, 'mechanisms': set()})
        }

    def run_all_cascades(self, max_depth: int = 3):
        """
        Run cascade analysis for all memory genes
        """
        print(f"\n{'='*80}")
        print(f"BULK CASCADE ANALYSIS: {len(self.memory_genes)} genes")
        print('='*80)

        successful = 0
        failed = []

        for i, gene in enumerate(self.memory_genes, 1):
            print(f"\n[{i}/{len(self.memory_genes)}] Analyzing {gene}...")

            try:
                analyzer = CRISPRCascadeAnalyzer(gene, self.organism, max_depth)
                cascade_data = analyzer.analyze()
                self.cascade_results[gene] = cascade_data
                successful += 1
            except Exception as e:
                print(f"  ✗ Failed to analyze {gene}: {e}")
                failed.append(gene)

        print(f"\n✅ Successfully analyzed {successful}/{len(self.memory_genes)} genes")
        if failed:
            print(f"❌ Failed genes: {', '.join(failed)}")

    def aggregate_intervention_points(self):
        """
        Find convergent intervention points across all genes
        """
        print(f"\n{'='*80}")
        print("AGGREGATING INTERVENTION POINTS")
        print('='*80)

        # Track all TFs and their properties
        tf_impact = defaultdict(lambda: {
            'regulated_genes': [],
            'pioneer_for': [],
            'amplifier_for': [],
            'signaling_for': [],
            'total_score': 0,
            'chromatin_opening_score': 0,
            'network_centrality': 0
        })

        # Process each gene's cascade
        for gene, cascade in self.cascade_results.items():
            # Process direct TFs
            for tf_data in cascade.get('direct_tfs', []):
                tf = tf_data['tf_name']
                tf_impact[tf]['regulated_genes'].append(gene)
                tf_impact[tf]['total_score'] += tf_data.get('binding_score', 1)

                # Classify by function
                if tf_data.get('tf_type') == 'pioneer':
                    if cascade.get('gene_chromatin', {}).get('healthy') == 'closed':
                        tf_impact[tf]['pioneer_for'].append(gene)
                        tf_impact[tf]['chromatin_opening_score'] += 10  # High value

                elif tf_data.get('tf_type') == 'settler':
                    tf_impact[tf]['amplifier_for'].append(gene)

                # Track signaling requirements
                if tf_data.get('requires_signal'):
                    for signal in tf_data['requires_signal']:
                        tf_impact[signal]['signaling_for'].append(f"{tf}->{gene}")

            # Process intervention strategies
            for strategy in cascade.get('strategies', {}).get('inhibition', []):
                target = strategy['target']
                if isinstance(target, str):
                    self.strategies['inhibition'][target]['genes'].append(gene)
                    self.strategies['inhibition'][target]['score'] += 1 / strategy.get('priority', 1)
                    self.strategies['inhibition'][target]['mechanisms'].add(strategy.get('mechanism', ''))

            for strategy in cascade.get('strategies', {}).get('activation', []):
                target = strategy['target']
                if isinstance(target, str):
                    self.strategies['activation'][target]['genes'].append(gene)
                    self.strategies['activation'][target]['score'] += 1 / strategy.get('priority', 1)
                    self.strategies['activation'][target]['mechanisms'].add(strategy.get('mechanism', ''))

        # Calculate network centrality
        for tf, data in tf_impact.items():
            data['network_centrality'] = len(data['regulated_genes'])

        self.master_regulators = dict(tf_impact)

    def identify_convergence_points(self):
        """
        Find TFs that appear in multiple regulatory cascades
        """
        print(f"\nIdentifying convergence points...")

        convergence_scores = {}

        for tf, data in self.master_regulators.items():
            # Multi-factor scoring
            score = 0

            # Coverage score (how many memory genes)
            coverage = len(data['regulated_genes'])
            score += coverage * 10

            # Pioneer score (chromatin opening capability)
            pioneer_genes = len(data['pioneer_for'])
            score += pioneer_genes * 50  # Highest weight for pioneers

            # Amplifier score
            amplifier_genes = len(data['amplifier_for'])
            score += amplifier_genes * 5

            # Signaling hub score
            signaling_targets = len(data['signaling_for'])
            score += signaling_targets * 3

            convergence_scores[tf] = {
                'total_score': score,
                'coverage': coverage,
                'pioneer_genes': pioneer_genes,
                'amplifier_genes': amplifier_genes,
                'signaling_targets': signaling_targets,
                'regulated_genes': data['regulated_genes'][:5]  # Sample for display
            }

        # Sort by score
        self.convergence_points = dict(
            sorted(convergence_scores.items(),
                   key=lambda x: x[1]['total_score'],
                   reverse=True)
        )

    def generate_master_strategy(self) -> Dict:
        """
        Generate prioritized master intervention strategy
        """
        print(f"\n{'='*80}")
        print("MASTER CRISPR STRATEGY")
        print('='*80)

        strategy = {
            'tier1_pioneers': [],
            'tier2_amplifiers': [],
            'tier3_signaling': [],
            'combinations': []
        }

        # Tier 1: Pioneer factors (highest priority)
        pioneers = [(tf, data) for tf, data in self.convergence_points.items()
                   if data['pioneer_genes'] > 0][:5]

        for tf, data in pioneers:
            strategy['tier1_pioneers'].append({
                'target': tf,
                'score': data['total_score'],
                'opens_chromatin_for': data['pioneer_genes'],
                'total_genes_affected': data['coverage'],
                'method': 'CRISPRi to prevent chromatin opening',
                'priority': 'HIGHEST'
            })

        # Tier 2: Amplifiers
        amplifiers = [(tf, data) for tf, data in self.convergence_points.items()
                     if data['amplifier_genes'] > 0 and data['pioneer_genes'] == 0][:5]

        for tf, data in amplifiers:
            strategy['tier2_amplifiers'].append({
                'target': tf,
                'score': data['total_score'],
                'amplifies_genes': data['amplifier_genes'],
                'total_genes_affected': data['coverage'],
                'method': 'CRISPRi to reduce expression',
                'priority': 'HIGH'
            })

        # Tier 3: Signaling components
        signaling = [(tf, data) for tf, data in self.convergence_points.items()
                    if data['signaling_targets'] > 2][:5]

        for tf, data in signaling:
            strategy['tier3_signaling'].append({
                'target': tf,
                'score': data['total_score'],
                'controls_pathways': data['signaling_targets'],
                'method': 'CRISPRi to block signaling',
                'priority': 'MODERATE'
            })

        # Combinations
        if pioneers and amplifiers:
            strategy['combinations'].append({
                'targets': [pioneers[0][0], amplifiers[0][0]],
                'rationale': 'Block both chromatin opening and expression amplification',
                'expected_synergy': 'HIGH'
            })

        return strategy

    def generate_report(self) -> str:
        """
        Generate comprehensive report
        """
        report = []
        report.append(f"\n{'='*80}")
        report.append("BULK CASCADE ANALYSIS REPORT")
        report.append('='*80)

        # Summary statistics
        report.append(f"\nAnalysis Summary:")
        report.append(f"  Memory genes analyzed: {len(self.cascade_results)}")
        report.append(f"  Total TFs identified: {len(self.master_regulators)}")
        report.append(f"  Convergence points: {len(self.convergence_points)}")

        # Top master regulators
        report.append(f"\nTop 10 Master Regulators:")
        report.append(f"{'Rank':<6} {'TF':<15} {'Score':<10} {'Coverage':<10} {'Pioneer':<10} {'Type'}")
        report.append("-"*70)

        for i, (tf, data) in enumerate(list(self.convergence_points.items())[:10], 1):
            tf_type = 'Pioneer' if data['pioneer_genes'] > 0 else 'Amplifier'
            report.append(f"{i:<6} {tf:<15} {data['total_score']:<10.0f} "
                         f"{data['coverage']:<10} {data['pioneer_genes']:<10} {tf_type}")

        # Pioneer factors
        pioneers = [(tf, data) for tf, data in self.convergence_points.items()
                   if data['pioneer_genes'] > 0]

        if pioneers:
            report.append(f"\nCritical Pioneer Factors (open closed chromatin):")
            for tf, data in pioneers[:5]:
                report.append(f"  {tf}: Opens chromatin for {data['pioneer_genes']} genes")
                report.append(f"    Example targets: {', '.join(data['regulated_genes'][:3])}")

        # Network hubs
        hubs = sorted(self.convergence_points.items(),
                     key=lambda x: x[1]['coverage'], reverse=True)[:5]

        report.append(f"\nNetwork Hubs (regulate many genes):")
        for tf, data in hubs:
            report.append(f"  {tf}: Regulates {data['coverage']} memory genes")

        # Strategy summary
        strategy = self.generate_master_strategy()

        report.append(f"\nRecommended CRISPR Strategy:")
        report.append(f"\nTier 1 - Pioneer Factors (HIGHEST PRIORITY):")
        for target in strategy['tier1_pioneers'][:3]:
            report.append(f"  🎯 {target['target']}: Prevents chromatin opening for {target['opens_chromatin_for']} genes")

        report.append(f"\nTier 2 - Expression Amplifiers:")
        for target in strategy['tier2_amplifiers'][:3]:
            report.append(f"  📈 {target['target']}: Reduces expression of {target['amplifies_genes']} genes")

        report.append(f"\nTier 3 - Signaling Components:")
        for target in strategy['tier3_signaling'][:3]:
            report.append(f"  🔌 {target['target']}: Blocks {target['controls_pathways']} signaling pathways")

        return '\n'.join(report)

    def save_results(self, output_dir: str = "results/bulk_cascade"):
        """
        Save aggregated results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save master regulators
        master_file = output_path / "master_regulators.json"
        with open(master_file, 'w') as f:
            json.dump(self.master_regulators, f, indent=2, default=str)

        # Save convergence points
        convergence_file = output_path / "convergence_points.json"
        with open(convergence_file, 'w') as f:
            json.dump(self.convergence_points, f, indent=2)

        # Save strategy
        strategy = self.generate_master_strategy()
        strategy_file = output_path / "master_strategy.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)

        # Save report
        report_file = output_path / "bulk_cascade_report.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_report())

        # Save CSV for analysis
        self.save_csv_summary(output_path)

        # Aggregate all TSV files for scoring
        self.aggregate_tsv_files(output_path)

        print(f"\n✅ Results saved to {output_dir}")

    def save_csv_summary(self, output_path: Path):
        """
        Save CSV summary for further analysis
        """
        # Create dataframe of master regulators
        rows = []
        for tf, data in self.convergence_points.items():
            rows.append({
                'TF': tf,
                'Total_Score': data['total_score'],
                'Coverage': data['coverage'],
                'Pioneer_Genes': data['pioneer_genes'],
                'Amplifier_Genes': data['amplifier_genes'],
                'Signaling_Targets': data['signaling_targets'],
                'Type': 'Pioneer' if data['pioneer_genes'] > 0 else 'Amplifier'
            })

        df = pd.DataFrame(rows)
        csv_file = output_path / "master_regulators.csv"
        df.to_csv(csv_file, index=False)

        print(f"  Saved CSV summary to {csv_file}")

    def aggregate_tsv_files(self, output_path: Path):
        """Aggregate all individual gene TSV files into one for scoring"""
        try:
            import pandas as pd

            # Find all TSV files from individual analyses
            tsv_files = []
            for gene in self.memory_genes:
                # Look for TSV files in the results directory structure
                tsv_path = Path(f"results/{gene}_analysis/{gene}_targets.tsv")
                if not tsv_path.exists():
                    # Try alternate location
                    tsv_path = output_path.parent / f"{gene}_analysis" / f"{gene}_targets.tsv"
                if tsv_path.exists():
                    tsv_files.append(tsv_path)

            if not tsv_files:
                print(f"  ⚠ No TSV files found to aggregate")
                return

            print(f"\n  Aggregating {len(tsv_files)} TSV files...")

            # Read and combine all TSV files
            dfs = []
            for tsv_file in tsv_files:
                try:
                    df = pd.read_csv(tsv_file, sep='\t')
                    dfs.append(df)
                except Exception as e:
                    print(f"    ⚠ Error reading {tsv_file.name}: {e}")

            if dfs:
                # Combine all dataframes
                combined_df = pd.concat(dfs, ignore_index=True)

                # Add impact column for crispr_score.py compatibility
                if 'impact' not in combined_df.columns and 'ism_max' in combined_df.columns:
                    combined_df['impact'] = combined_df['ism_max']

                # Save combined file
                combined_file = output_path / "all_targets_combined.tsv"
                combined_df.to_csv(combined_file, sep='\t', index=False)

                print(f"  ✓ Aggregated {len(combined_df)} targets from {len(dfs)} genes")
                print(f"  📁 Combined TSV saved to {combined_file}")
                print(f"\n  Ready for scoring: python crispr_score.py {combined_file}")

        except Exception as e:
            print(f"  ⚠ Could not aggregate TSV files: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='CRISPR Bulk Aggregator - Find master regulators across gene sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze default memory genes
  python crispr_bulk_aggregator.py

  # Use custom gene list
  python crispr_bulk_aggregator.py --genes IL1B IL6 TNF CCL2 GFAP

  # Load genes from file
  python crispr_bulk_aggregator.py --gene-file memory_genes.txt

  # Set cascade depth
  python crispr_bulk_aggregator.py --depth 4
        '''
    )

    parser.add_argument('--genes', nargs='+',
                       help='List of genes to analyze')
    parser.add_argument('--gene-file',
                       help='File with gene list (one per line)')
    parser.add_argument('--organism', default='human',
                       choices=['human', 'mouse'],
                       help='Organism (default: human)')
    parser.add_argument('--depth', type=int, default=3,
                       help='Maximum cascade depth (default: 3)')
    parser.add_argument('--output', default='results/bulk_cascade',
                       help='Output directory')

    args = parser.parse_args()

    # Determine gene list
    if args.genes:
        memory_genes = args.genes
    elif args.gene_file:
        with open(args.gene_file) as f:
            memory_genes = [line.strip() for line in f if line.strip()]
    else:
        # Default astrocyte memory genes
        memory_genes = ['IL1B', 'IL6', 'TNF', 'CCL2', 'CCL5', 'GFAP', 'VIM', 'STAT3']

    # Run aggregation
    aggregator = CRISPRBulkAggregator(memory_genes, args.organism)

    # Run all cascades
    aggregator.run_all_cascades(max_depth=args.depth)

    # Aggregate results
    aggregator.aggregate_intervention_points()
    aggregator.identify_convergence_points()

    # Generate and display report
    print(aggregator.generate_report())

    # Save results
    aggregator.save_results(args.output)

    print(f"\n✅ Bulk cascade analysis complete!")
    print(f"   Top target: {list(aggregator.convergence_points.keys())[0] if aggregator.convergence_points else 'None'}")


if __name__ == '__main__':
    main()