#!/usr/bin/env python3
"""
Co-perturbation validation testing using CellOracle
Tests predictions from ISM analysis against co-targeting strategies

-----------------------------------------------------------------------
Copyright (c) 2025 Sean Kiewiet. All rights reserved.
-----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple
import json
from datetime import datetime
from tf_network_api import TFNetworkAPI

class CoperturbationValidator:
    """Test co-perturbation strategies based on ISM predictions"""

    def __init__(self):
        """Initialize with validated genes and ISM scores"""
        # Top validated genes from Leng et al. with ISM scores
        self.validated_genes = {
            'JUN': 146539,    # AP-1 complex
            'FOS': 143142,    # AP-1 complex
            'RELA': 106360,   # NF-κB p65
            'STAT3': 99117,   # JAK-STAT
            'CEBPB': 61275,   # C/EBP TF
            'NFKB1': 42539    # NF-κB p50
        }

        # Load GRN for network analysis
        with open('merged_grn_promoter_trrust.pkl', 'rb') as f:
            self.grn = pickle.load(f)

        # Initialize TF network API
        self.api = TFNetworkAPI()

        # Store results
        self.results = {
            'single_perturbations': {},
            'co_perturbations': {},
            'synergy_scores': {},
            'compensation_analysis': {}
        }

    def analyze_single_perturbation(self, gene: str) -> Dict:
        """Analyze effect of single gene perturbation"""
        print(f"\nAnalyzing single perturbation: {gene}")

        # Get downstream targets
        targets = self.api.get_targets(gene)

        # Get regulators (for compensation analysis)
        regulators = self.api.get_regulators(gene)

        # Calculate network impact
        cascade = self.api.get_cascade(gene, max_depth=2)
        total_affected = sum(len(nodes) for nodes in cascade.values())

        result = {
            'gene': gene,
            'ism_score': self.validated_genes.get(gene, 0),
            'direct_targets': len(targets),
            'targets_list': targets[:10],  # Top 10
            'regulators': len(regulators),
            'cascade_size': total_affected,
            'cascade_depth_1': len(cascade.get('depth_1', [])),
            'cascade_depth_2': len(cascade.get('depth_2', []))
        }

        # Check for key inflammatory genes affected
        inflammatory_genes = ['IL1B', 'IL6', 'TNF', 'CCL2', 'ICAM1', 'VCAM1']
        affected_inflammatory = [g for g in inflammatory_genes if g in targets]
        result['inflammatory_targets'] = affected_inflammatory

        return result

    def analyze_co_perturbation(self, genes: List[str], name: str) -> Dict:
        """Analyze effect of co-perturbation"""
        print(f"\nAnalyzing co-perturbation: {name} ({', '.join(genes)})")

        # Combined targets (union)
        all_targets = set()
        shared_targets = None

        for gene in genes:
            targets = set(self.api.get_targets(gene))
            all_targets.update(targets)

            if shared_targets is None:
                shared_targets = targets
            else:
                shared_targets = shared_targets.intersection(targets)

        # Combined cascade
        all_cascade_nodes = set()
        for gene in genes:
            cascade = self.api.get_cascade(gene, max_depth=2)
            for nodes in cascade.values():
                all_cascade_nodes.update(nodes)

        result = {
            'name': name,
            'genes': genes,
            'combined_ism': sum(self.validated_genes.get(g, 0) for g in genes),
            'total_targets': len(all_targets),
            'shared_targets': len(shared_targets),
            'cascade_size': len(all_cascade_nodes)
        }

        # Check inflammatory genes
        inflammatory_genes = ['IL1B', 'IL6', 'TNF', 'CCL2', 'ICAM1', 'VCAM1']
        affected = [g for g in inflammatory_genes if g in all_targets]
        result['inflammatory_targets'] = affected

        # Check for cross-regulation
        cross_regulation = []
        for g1 in genes:
            for g2 in genes:
                if g1 != g2:
                    if g2 in self.api.get_targets(g1):
                        cross_regulation.append(f"{g1}→{g2}")
        result['cross_regulation'] = cross_regulation

        return result

    def calculate_synergy(self, genes: List[str], co_result: Dict) -> float:
        """Calculate synergy score for co-perturbation"""
        # Sum of individual effects
        individual_sum = 0
        for gene in genes:
            if gene in self.results['single_perturbations']:
                individual_sum += self.results['single_perturbations'][gene]['cascade_size']

        # Combined effect
        combined_effect = co_result['cascade_size']

        # Synergy = (combined - sum) / sum
        if individual_sum > 0:
            synergy = (combined_effect - individual_sum) / individual_sum
        else:
            synergy = 0

        return synergy

    def analyze_compensation(self, perturbed_genes: List[str]) -> Dict:
        """Analyze potential compensation mechanisms"""
        print(f"\nAnalyzing compensation for: {', '.join(perturbed_genes)}")

        compensation = {
            'perturbed': perturbed_genes,
            'potential_compensators': {}
        }

        # For each perturbed gene, find TFs that could compensate
        for gene in perturbed_genes:
            # Get targets of this gene
            targets = self.api.get_targets(gene)

            # For each target, find other regulators that could compensate
            for target in targets[:10]:  # Check top 10 targets
                other_regulators = self.api.get_regulators(target)
                # Remove the perturbed genes
                compensators = [tf for tf in other_regulators
                               if tf not in perturbed_genes]

                if compensators:
                    compensation['potential_compensators'][target] = compensators[:5]

        return compensation

    def test_pathway_interactions(self) -> Dict:
        """Test interactions between AP-1 and NF-κB pathways"""
        print("\nTesting pathway interactions...")

        interactions = {}

        # AP-1 → NF-κB
        ap1_to_nfkb = []
        for ap1 in ['JUN', 'FOS']:
            for nfkb in ['RELA', 'NFKB1']:
                if nfkb in self.api.get_targets(ap1):
                    ap1_to_nfkb.append(f"{ap1}→{nfkb}")

        # NF-κB → AP-1
        nfkb_to_ap1 = []
        for nfkb in ['RELA', 'NFKB1']:
            for ap1 in ['JUN', 'FOS']:
                if ap1 in self.api.get_targets(nfkb):
                    nfkb_to_ap1.append(f"{nfkb}→{ap1}")

        interactions['AP1_to_NFkB'] = ap1_to_nfkb
        interactions['NFkB_to_AP1'] = nfkb_to_ap1

        # Shared targets
        ap1_targets = set()
        for gene in ['JUN', 'FOS']:
            ap1_targets.update(self.api.get_targets(gene))

        nfkb_targets = set()
        for gene in ['RELA', 'NFKB1']:
            nfkb_targets.update(self.api.get_targets(gene))

        shared = ap1_targets.intersection(nfkb_targets)
        interactions['shared_targets'] = len(shared)
        interactions['shared_examples'] = list(shared)[:10]

        return interactions

    def run_validation(self):
        """Run complete validation analysis"""
        print("="*80)
        print("CO-PERTURBATION VALIDATION ANALYSIS")
        print("="*80)
        print(f"Testing {len(self.validated_genes)} validated genes from Leng et al.")

        # 1. Test single perturbations
        print("\n" + "="*60)
        print("PHASE 1: SINGLE PERTURBATIONS")
        print("="*60)

        for gene in self.validated_genes:
            result = self.analyze_single_perturbation(gene)
            self.results['single_perturbations'][gene] = result

        # 2. Test co-perturbations
        print("\n" + "="*60)
        print("PHASE 2: CO-PERTURBATIONS")
        print("="*60)

        co_perturbation_sets = {
            'AP-1_complex': ['JUN', 'FOS'],
            'NFkB_complex': ['RELA', 'NFKB1'],
            'AP1_NFkB_dual': ['JUN', 'RELA'],
            'Triple_max': ['JUN', 'FOS', 'RELA'],
            'STAT3_CEBPB': ['STAT3', 'CEBPB'],
            'Full_validated': ['JUN', 'FOS', 'RELA', 'STAT3', 'CEBPB', 'NFKB1']
        }

        for name, genes in co_perturbation_sets.items():
            result = self.analyze_co_perturbation(genes, name)
            self.results['co_perturbations'][name] = result

            # Calculate synergy
            synergy = self.calculate_synergy(genes, result)
            self.results['synergy_scores'][name] = synergy

        # 3. Analyze compensation
        print("\n" + "="*60)
        print("PHASE 3: COMPENSATION ANALYSIS")
        print("="*60)

        compensation_tests = {
            'AP1_only': ['JUN', 'FOS'],
            'NFkB_only': ['RELA', 'NFKB1'],
            'Dual_block': ['JUN', 'FOS', 'RELA', 'NFKB1']
        }

        for name, genes in compensation_tests.items():
            comp = self.analyze_compensation(genes)
            self.results['compensation_analysis'][name] = comp

        # 4. Test pathway interactions
        self.results['pathway_interactions'] = self.test_pathway_interactions()

        # 5. Generate report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'coperturbation_validation_{timestamp}.txt'

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CO-PERTURBATION VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Single perturbations
            f.write("="*60 + "\n")
            f.write("1. SINGLE PERTURBATION EFFECTS\n")
            f.write("="*60 + "\n\n")

            f.write(f"{'Gene':<10} {'ISM Score':<12} {'Targets':<10} {'Cascade':<10} {'Inflammatory':<30}\n")
            f.write("-"*70 + "\n")

            for gene, data in sorted(self.results['single_perturbations'].items(),
                                    key=lambda x: x[1]['ism_score'], reverse=True):
                inflam = ', '.join(data['inflammatory_targets'])
                f.write(f"{gene:<10} {data['ism_score']:<12,} {data['direct_targets']:<10} "
                       f"{data['cascade_size']:<10} {inflam:<30}\n")

            # Co-perturbations
            f.write("\n" + "="*60 + "\n")
            f.write("2. CO-PERTURBATION EFFECTS\n")
            f.write("="*60 + "\n\n")

            f.write(f"{'Strategy':<20} {'Combined ISM':<15} {'Targets':<10} {'Cascade':<10} {'Synergy':<10}\n")
            f.write("-"*70 + "\n")

            for name, data in sorted(self.results['co_perturbations'].items(),
                                    key=lambda x: x[1]['combined_ism'], reverse=True):
                synergy = self.results['synergy_scores'].get(name, 0)
                f.write(f"{name:<20} {data['combined_ism']:<15,} {data['total_targets']:<10} "
                       f"{data['cascade_size']:<10} {synergy:+.2%}\n")

            # Key findings
            f.write("\n" + "="*60 + "\n")
            f.write("3. KEY FINDINGS\n")
            f.write("="*60 + "\n\n")

            # Best single vs best combo
            best_single = max(self.results['single_perturbations'].values(),
                            key=lambda x: x['cascade_size'])
            best_combo = max(self.results['co_perturbations'].values(),
                           key=lambda x: x['cascade_size'])

            f.write(f"BEST SINGLE TARGET:\n")
            f.write(f"  {best_single['gene']}: {best_single['cascade_size']} nodes affected\n\n")

            f.write(f"BEST CO-PERTURBATION:\n")
            f.write(f"  {best_combo['name']}: {best_combo['cascade_size']} nodes affected\n")
            f.write(f"  Genes: {', '.join(best_combo['genes'])}\n\n")

            # Synergy analysis
            f.write("SYNERGY ANALYSIS:\n")
            positive_synergy = {k: v for k, v in self.results['synergy_scores'].items() if v > 0}
            if positive_synergy:
                for name, score in sorted(positive_synergy.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {name}: {score:+.1%} synergy\n")

            # Pathway interactions
            f.write("\n" + "="*60 + "\n")
            f.write("4. PATHWAY CROSS-TALK\n")
            f.write("="*60 + "\n\n")

            interactions = self.results['pathway_interactions']
            f.write(f"AP-1 → NF-κB: {interactions['AP1_to_NFkB']}\n")
            f.write(f"NF-κB → AP-1: {interactions['NFkB_to_AP1']}\n")
            f.write(f"Shared targets: {interactions['shared_targets']} genes\n")
            f.write(f"Examples: {', '.join(interactions['shared_examples'][:5])}\n")

            # Recommendations
            f.write("\n" + "="*60 + "\n")
            f.write("5. RECOMMENDATIONS\n")
            f.write("="*60 + "\n\n")

            f.write("Based on network analysis:\n\n")

            # Find best strategies
            strategies = []
            for name, data in self.results['co_perturbations'].items():
                efficiency = data['cascade_size'] / data['combined_ism'] * 1000000 if data['combined_ism'] > 0 else 0
                strategies.append({
                    'name': name,
                    'genes': data['genes'],
                    'efficiency': efficiency,
                    'cascade': data['cascade_size'],
                    'ism': data['combined_ism']
                })

            strategies.sort(key=lambda x: x['efficiency'], reverse=True)

            f.write("TOP STRATEGIES BY EFFICIENCY (cascade/ISM):\n")
            for i, s in enumerate(strategies[:3], 1):
                f.write(f"{i}. {s['name']}: {s['efficiency']:.2f} efficiency\n")
                f.write(f"   Targets {s['cascade']} nodes with ISM {s['ism']:,}\n")
                f.write(f"   Genes: {', '.join(s['genes'])}\n\n")

        print(f"\nReport saved to: {report_file}")

        # Also save JSON for programmatic access
        json_file = f'coperturbation_validation_{timestamp}.json'
        with open(json_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            json_results = self.results.copy()
            for key in json_results:
                if isinstance(json_results[key], dict):
                    for subkey in json_results[key]:
                        if isinstance(json_results[key][subkey], set):
                            json_results[key][subkey] = list(json_results[key][subkey])
            json.dump(json_results, f, indent=2)

        print(f"JSON data saved to: {json_file}")


def main():
    """Run co-perturbation validation"""
    validator = CoperturbationValidator()
    validator.run_validation()

    # Close API connection
    validator.api.close()


if __name__ == '__main__':
    main()