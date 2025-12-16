#!/usr/bin/env python3
"""
CRISPR Cascade Analyzer - Evidence-Based Target Discovery

Core workhorse for identifying optimal CRISPR targets based on:
1. TF regulatory networks (CellOracle)
2. Chromatin accessibility (AlphaGenome)
3. ISM impact scoring (AlphaGenome)
4. Network topology analysis

Designed for reliability and bulk processing compatibility.
"""

import json
import sys
import requests
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class CRISPRCascadeAnalyzer:
    """
    Evidence-based CRISPR target analyzer
    """

    def __init__(self, gene: str, organism: str = 'human',
                 max_depth: int = 3, ism_bps: int = 100,
                 ism_depth: int = 2, strategy: str = 'auto',
                 healthy_ontology: str = None, pathologic_ontology: str = None):
        """
        Initialize analyzer with evidence-based scoring

        Args:
            gene: Gene symbol to analyze
            organism: Species (human/mouse)
            max_depth: Cascade depth to trace
            ism_bps: ISM region size in bp
            ism_depth: 1=gene only, 2=gene+TFs
            strategy: 'inhibit', 'activate', or 'auto'
            healthy_ontology: Cell type ontology for healthy state (default: CL:0000127 for astrocyte)
            pathologic_ontology: Cell type ontology for pathologic state (default: same as healthy)
        """
        self.gene = gene.upper()
        self.organism = organism
        self.max_depth = max_depth
        self.ism_bps = ism_bps
        self.ism_depth = ism_depth
        self.strategy = strategy

        # Set default ontologies
        self.healthy_ontology = healthy_ontology or 'CL:0000127'  # Default: astrocyte
        self.pathologic_ontology = pathologic_ontology or self.healthy_ontology

        # Evidence collection
        self.evidence = {
            'gene_info': {},
            'direct_tfs': [],
            'chromatin_state': {},
            'ism_scores': {},
            'network_topology': {},
            'targeting_evidence': {},
            'optimal_coordinates': None,
            'confidence_score': 0.0
        }

        # Scoring weights for evidence-based decision
        self.scoring_weights = {
            'ism_impact': 0.35,      # ISM mutation impact
            'network_specificity': 0.25,  # Network topology
            'chromatin_accessibility': 0.20,  # Chromatin state
            'tf_binding': 0.10,      # TF binding strength
            'distance_to_tss': 0.10  # Proximity to TSS
        }

        # Results directory
        self.results_dir = Path(f"results/{self.gene}_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self) -> Dict:
        """
        Main analysis pipeline with evidence collection
        Returns evidence-based targeting recommendations
        """
        print(f"\n{'='*80}")
        print(f"EVIDENCE-BASED CRISPR ANALYSIS: {self.gene}")
        print(f"{'='*80}")

        # Step 1: Gene information
        self._collect_gene_evidence()

        # Step 2: TF regulatory network
        self._collect_tf_evidence()

        # Step 3: Chromatin accessibility
        self._collect_chromatin_evidence()

        # Step 4: ISM impact scoring
        self._collect_ism_evidence()

        # Step 5: Network topology analysis
        self._analyze_network_topology()

        # Step 6: Generate targeting strategy
        self._generate_targeting_strategy()

        # Step 7: Calculate confidence score
        self._calculate_confidence_score()

        # Step 8: Save evidence report
        self._save_evidence_report()

        return self.evidence

    def _collect_gene_evidence(self):
        """Collect gene information from Ensembl"""
        print(f"\n[1/8] Collecting gene evidence from Ensembl...")

        try:
            # Query Ensembl API
            url = f"https://rest.ensembl.org/lookup/symbol/{self.organism}/{self.gene}?"
            r = requests.get(url, headers={"Content-Type": "application/json"})

            if r.ok:
                data = r.json()
                self.evidence['gene_info'] = {
                    'symbol': self.gene,
                    'chromosome': f"chr{data.get('seq_region_name')}",
                    'start': data.get('start'),
                    'end': data.get('end'),
                    'strand': data.get('strand'),
                    'tss': data.get('start') if data.get('strand') > 0 else data.get('end'),
                    'biotype': data.get('biotype'),
                    'ensembl_id': data.get('id'),
                    'location': f"chr{data.get('seq_region_name')}:{data.get('start')}-{data.get('end')}"
                }
                print(f"  ✓ Gene location: {self.evidence['gene_info']['location']}")
                print(f"  ✓ TSS: {self.evidence['gene_info']['tss']:,}")
                print(f"  ✓ Biotype: {self.evidence['gene_info']['biotype']}")
            else:
                print(f"  ⚠ Gene {self.gene} not found in Ensembl")
                self.evidence['gene_info'] = {'error': 'Gene not found'}

        except Exception as e:
            print(f"  ✗ Ensembl query failed: {e}")
            self.evidence['gene_info'] = {'error': str(e)}

    def _collect_tf_evidence(self):
        """Collect TF regulatory evidence from CellOracle"""
        print(f"\n[2/8] Collecting TF regulatory evidence...")

        try:
            # Use absolute path to tf_query_grn.py
            tf_query_path = '/Users/seankiewiet/dev/projects/bio/CellOracle/tf_query_grn.py'

            # Check if file exists
            import os
            if not os.path.exists(tf_query_path):
                print(f"\n❌ CRITICAL ERROR: CellOracle tf_query_grn.py not found at {tf_query_path}")
                print("   This tool is required for TF network analysis.")
                print("   Please ensure CellOracle is properly installed.")
                sys.exit(1)

            # Run from CellOracle directory to access the database
            celloracle_dir = '/Users/seankiewiet/dev/projects/bio/CellOracle'
            cmd = [tf_query_path, self.gene, '--raw']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=celloracle_dir)

            if result.returncode == 0 and result.stdout:
                try:
                    data = json.loads(result.stdout)
                    regulators = data.get('regulators', [])
                except json.JSONDecodeError as e:
                    print(f"\n❌ CRITICAL ERROR: Invalid response from CellOracle")
                    print(f"   Response: {result.stdout[:200]}")
                    print(f"   Error: {e}")
                    sys.exit(1)

                if not regulators:
                    print(f"\n❌ CRITICAL ERROR: No TF regulators found for {self.gene}")
                    print("   This gene may not be in the CellOracle database or may not have known regulators.")
                    print("   Cannot proceed with CRISPR target analysis without regulatory information.")
                    sys.exit(1)

                print(f"  ✓ Found {len(regulators)} TF regulators")

                # Store TF evidence
                self.evidence['direct_tfs'] = []
                for tf in regulators[:20]:  # Top 20 TFs
                    tf_data = {
                        'tf_name': tf,
                        'binding_score': 1.0,  # Default score
                        'ism_score': 0.0,       # Will be filled by ISM
                        'targets': 0,           # Will be filled by topology
                        'evidence_type': 'CellOracle'
                    }
                    self.evidence['direct_tfs'].append(tf_data)

                # Display top TFs
                if regulators:
                    print(f"  Top regulators: {', '.join(regulators[:5])}")
            else:
                print(f"\n❌ CRITICAL ERROR: CellOracle query failed")
                print(f"   Command: {' '.join(cmd)}")
                print(f"   Return code: {result.returncode}")
                print(f"   Error: {result.stderr[:500] if result.stderr else 'No error output'}")
                print("   Cannot proceed without TF regulatory information.")
                sys.exit(1)

        except subprocess.CalledProcessError as e:
            print(f"\n❌ CRITICAL ERROR: Failed to execute CellOracle query")
            print(f"   Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Unexpected error querying CellOracle")
            print(f"   Error: {e}")
            sys.exit(1)

    def _collect_chromatin_evidence(self):
        """Collect chromatin accessibility evidence from AlphaGenome"""
        print(f"\n[3/8] Identifying regulatory regions using AlphaGenome...")
        print(f"  Evidence source: AlphaGenome chromatin accessibility predictions")

        if not self.evidence['gene_info'].get('tss'):
            print("  ⚠ No TSS information available")
            return

        try:
            from AlphaGenome import AlphaGenome

            ag = AlphaGenome()
            tss = self.evidence['gene_info']['tss']
            chrom = self.evidence['gene_info']['chromosome']

            # Initialize regulatory regions storage
            self.evidence['regulatory_regions'] = {
                'promoter': None,
                'enhancers': []
            }

            # 1. Identify PROMOTER region (typically -500 to +100 from TSS)
            promoter_start = tss - 500
            promoter_end = tss + 100
            print(f"\n  Analyzing PROMOTER region: {chrom}:{promoter_start}-{promoter_end}")
            print(f"    Cell type: {self.healthy_ontology} (astrocyte)")

            # AlphaGenome only predicts for ONE cell type at a time
            promoter_result = self._predict_chromatin_state(
                chrom, promoter_start, promoter_end, 'healthy'
            )

            if promoter_result:
                self.evidence['regulatory_regions']['promoter'] = {
                    'coordinates': f"{chrom}:{promoter_start}-{promoter_end}",
                    'start': promoter_start,
                    'end': promoter_end,
                    'type': 'promoter',
                    'chromatin_state': promoter_result['state'],
                    'dnase_signal': promoter_result['signal'],
                    'cell_type': self.healthy_ontology
                }
                print(f"    ✓ Promoter chromatin: {promoter_result['state']} [DNase={promoter_result['signal']:.3f}]")

            # 2. Scan for ENHANCER regions (typically within 10kb upstream)
            # Check multiple windows upstream for open chromatin regions
            print(f"\n  Scanning for ENHANCER regions (up to 10kb upstream)...")

            enhancer_windows = [
                (tss - 2000, tss - 1000, 'proximal_enhancer'),
                (tss - 5000, tss - 4000, 'distal_enhancer_1'),
                (tss - 10000, tss - 9000, 'distal_enhancer_2')
            ]

            enhancer_candidates = []  # Store all tested regions for reporting
            dnase_threshold = 0.3  # Threshold for considering region as active enhancer

            for window_start, window_end, window_name in enhancer_windows:
                print(f"\n    Testing {window_name}: {chrom}:{window_start}-{window_end}")
                print(f"      Distance from TSS: {abs(window_start - tss):,}bp")

                # Test each potential enhancer window
                result = self._predict_chromatin_state(
                    chrom, window_start, window_end, 'healthy'
                )

                if result:
                    print(f"      Healthy DNase signal: {result['signal']:.3f} ({'PASS' if result['signal'] > dnase_threshold else 'BELOW THRESHOLD'})")

                    # Store candidate info even if below threshold
                    candidate_info = {
                        'coordinates': f"{chrom}:{window_start}-{window_end}",
                        'start': window_start,
                        'end': window_end,
                        'type': window_name,
                        'healthy_state': result['state'],
                        'healthy_dnase': result['signal'],
                        'distance_to_tss': abs(window_start - tss),
                        'passed_threshold': result['signal'] > dnase_threshold
                    }

                    if result['signal'] > dnase_threshold:
                        # Add to enhancers list if passes threshold
                        self.evidence['regulatory_regions']['enhancers'].append(candidate_info)
                        print(f"    ✓ ENHANCER IDENTIFIED: {window_name}")
                    else:
                        print(f"      ⚠ Not considered enhancer (DNase < {dnase_threshold})")

                    enhancer_candidates.append(candidate_info)

            # Report summary
            if not self.evidence['regulatory_regions']['enhancers']:
                print(f"\n    SUMMARY: No enhancers detected (0/{len(enhancer_candidates)} regions passed DNase threshold > {dnase_threshold})")
            else:
                print(f"\n    SUMMARY: {len(self.evidence['regulatory_regions']['enhancers'])} enhancer(s) identified")

            # Store summary for compatibility
            if promoter_result:
                self.evidence['chromatin_state'] = {
                    'state': promoter_result['state'],
                    'dnase_signal': promoter_result['signal'],
                    'accessibility': 'open' if promoter_result['signal'] > 0.5 else 'closed',
                    'evidence_source': 'AlphaGenome DNase-seq predictions',
                    'test_region': f"{chrom}:{promoter_start}-{promoter_end}",
                    'cell_type': self.healthy_ontology,
                    'note': 'AlphaGenome predicts for single cell type only'
                }

        except Exception as e:
            print(f"  ✗ AlphaGenome query failed: {e}")
            self.evidence['chromatin_state'] = {'error': str(e)}

    def _collect_ism_evidence(self):
        """Collect ISM impact evidence for identified regulatory regions"""
        print(f"\n[4/8] Collecting ISM impact evidence for regulatory regions...")
        print(f"  Evidence source: AlphaGenome ISM (In-Silico Mutagenesis) scoring")
        print(f"  Method: L2 norm of mutation impacts across 20 scorer models")

        if not self.evidence.get('regulatory_regions'):
            # Fallback to TSS-based analysis if no regions identified
            if not self.evidence['gene_info'].get('tss'):
                print("  ⚠ No TSS information available")
                return

            tss = self.evidence['gene_info']['tss']
            chrom = self.evidence['gene_info']['chromosome']

            print(f"\n  Fallback: Testing proximal promoter region")
            promoter_start = tss - 50
            promoter_end = tss + 50

            print(f"    Testing region: {chrom}:{promoter_start}-{promoter_end} ({self.ism_bps}bp window)")
            print(f"    Rationale: Proximal promoter region typically contains critical regulatory elements")

            gene_ism = self._run_ism_analysis(
                chrom=chrom,
                center=tss,
                window=self.ism_bps,
                target_name=f"{self.gene}_promoter"
            )

            if gene_ism:
                self.evidence['ism_scores']['gene_enhancer'] = gene_ism
                print(f"    ✓ Gene enhancer ISM max impact: {gene_ism['max_impact']:.2f}")
                print(f"    ✓ Mean impact across {gene_ism['n_mutations']} mutations: {gene_ism['mean_impact']:.2f}")
            return

        chrom = self.evidence['gene_info']['chromosome']

        # Test PROMOTER region if identified
        if self.evidence['regulatory_regions'].get('promoter'):
            promoter = self.evidence['regulatory_regions']['promoter']
            print(f"\n  Testing PROMOTER region: {promoter['coordinates']}")
            print(f"    Rationale: Core promoter contains TATA box and transcription start site")

            # Center ISM on middle of promoter
            promoter_center = (promoter['start'] + promoter['end']) // 2

            promoter_ism = self._run_ism_analysis(
                chrom=chrom,
                center=promoter_center,
                window=min(self.ism_bps, promoter['end'] - promoter['start']),
                target_name=f"{self.gene}_promoter"
            )

            if promoter_ism:
                self.evidence['ism_scores']['promoter'] = promoter_ism
                print(f"    ✓ PROMOTER ISM max impact: {promoter_ism['max_impact']:.2f}")
                print(f"    ✓ Mean impact: {promoter_ism['mean_impact']:.2f} ({promoter_ism['n_mutations']} mutations)")

        # Test ENHANCER regions if identified
        for i, enhancer in enumerate(self.evidence['regulatory_regions'].get('enhancers', [])):
            print(f"\n  Testing ENHANCER region: {enhancer['coordinates']}")
            print(f"    Type: {enhancer['type']} ({enhancer['distance_to_tss']}bp from TSS)")
            print(f"    Rationale: Open chromatin region with potential regulatory activity")

            # Center ISM on middle of enhancer
            enhancer_center = (enhancer['start'] + enhancer['end']) // 2

            enhancer_ism = self._run_ism_analysis(
                chrom=chrom,
                center=enhancer_center,
                window=min(self.ism_bps, enhancer['end'] - enhancer['start']),
                target_name=f"{self.gene}_enhancer_{i}"
            )

            if enhancer_ism:
                self.evidence['ism_scores'][f'enhancer_{i}'] = enhancer_ism
                print(f"    ✓ ENHANCER ISM max impact: {enhancer_ism['max_impact']:.2f}")
                print(f"    ✓ Mean impact: {enhancer_ism['mean_impact']:.2f} ({enhancer_ism['n_mutations']} mutations)")

        # For backward compatibility, keep gene_enhancer as highest scoring region
        best_score = 0
        best_region = None
        for key, ism_data in self.evidence['ism_scores'].items():
            if ism_data['max_impact'] > best_score:
                best_score = ism_data['max_impact']
                best_region = key

        if best_region and best_region in self.evidence['ism_scores']:
            self.evidence['ism_scores']['gene_enhancer'] = self.evidence['ism_scores'][best_region]

        # Level 2: TF enhancer ISM (if requested)
        if self.ism_depth >= 2 and self.evidence['direct_tfs']:
            print(f"\n  Level 2: TF binding site ISM...")
            print(f"    Testing upstream regulatory regions for TF binding sites")

            tss = self.evidence['gene_info']['tss']
            chrom = self.evidence['gene_info']['chromosome']

            for tf_data in self.evidence['direct_tfs'][:3]:  # Top 3 TFs
                tf_name = tf_data['tf_name']

                # Test region upstream of TSS where TF binding sites typically occur
                tf_center = tss - 100
                tf_start = tf_center - 25
                tf_end = tf_center + 25

                print(f"\n    Testing {tf_name} binding region: {chrom}:{tf_start}-{tf_end}")
                print(f"    Rationale: Common TF binding location upstream of TSS")

                # Run ISM for TF binding sites near gene
                tf_ism = self._run_ism_analysis(
                    chrom=chrom,
                    center=tf_center,
                    window=50,
                    target_name=f"{tf_name}_binding_{self.gene}"
                )

                if tf_ism:
                    tf_data['ism_score'] = tf_ism['max_impact']
                    print(f"    ✓ {tf_name} binding ISM: {tf_ism['max_impact']:.2f}")

    def _run_ism_analysis(self, chrom: str, center: int, window: int,
                          target_name: str) -> Optional[Dict]:
        """Run ISM analysis on a genomic region"""
        try:
            from AlphaGenome import AlphaGenome

            ag = AlphaGenome()

            # Define ISM region
            ism_start = center - window // 2
            ism_end = center + window // 2

            # Run ISM
            result = ag.score_ism_variants(
                chromosome=chrom,
                ism_start=int(ism_start),
                ism_end=int(ism_end),
                organism=self.organism,
                run_name=target_name,
                output_dir=str(self.results_dir)
            )

            if result and 'output_file' in result:
                # Parse ISM results
                npz_file = result['output_file']
                json_file = npz_file.replace('.npz', '.json')

                with open(json_file) as f:
                    metadata = json.load(f)

                ism_data = np.load(npz_file, allow_pickle=True)

                # Calculate impact scores
                impacts = []
                for mut in metadata.get('mutations', []):
                    mut_idx = mut['mutation_idx']
                    total_impact = 0
                    scorer_count = 0

                    for i in range(20):  # Check up to 20 scorers
                        key = f'mutation_{mut_idx}_scorer_{i}'
                        if key in ism_data:
                            data = ism_data[key]
                            if data.size > 0:
                                impact = np.sqrt(np.sum(data**2))  # L2 norm
                                if not np.isnan(impact):
                                    total_impact += impact
                                    scorer_count += 1

                    if scorer_count > 0:
                        impacts.append(total_impact / scorer_count)

                if impacts:
                    return {
                        'chromosome': chrom,
                        'start': ism_start,
                        'end': ism_end,
                        'max_impact': float(np.max(impacts)),
                        'mean_impact': float(np.mean(impacts)),
                        'std_impact': float(np.std(impacts)),
                        'n_mutations': len(impacts),
                        'high_impact_positions': self._find_high_impact_positions(
                            impacts, metadata['mutations'], threshold=0.8
                        )
                    }

        except Exception as e:
            print(f"    ⚠ ISM failed for {target_name}: {e}")

        return None

    def _find_high_impact_positions(self, impacts: List[float],
                                   mutations: List[Dict],
                                   threshold: float = 0.8) -> List[Dict]:
        """Find positions with highest ISM impact"""
        impacts = np.array(impacts)
        threshold_value = np.percentile(impacts, threshold * 100)

        high_impact = []
        for i, impact in enumerate(impacts):
            if impact >= threshold_value:
                mut = mutations[i]
                variant_parts = mut['variant'].split(':')
                if len(variant_parts) >= 2:
                    position = int(variant_parts[1])
                    high_impact.append({
                        'position': position,
                        'impact': float(impact),
                        'variant': mut['variant']
                    })

        return sorted(high_impact, key=lambda x: x['impact'], reverse=True)[:10]

    def _analyze_network_topology(self):
        """Analyze network topology for targeting decisions"""
        print(f"\n[5/8] Analyzing network topology...")

        topology = {
            'gene_in_degree': len(self.evidence['direct_tfs']),
            'gene_out_degree': 0,  # Genes typically don't regulate others
            'targeting_recommendation': None,
            'rationale': None
        }

        # Analyze each TF's network impact
        for tf_data in self.evidence['direct_tfs'][:5]:
            tf_name = tf_data['tf_name']

            # Get TF's downstream targets from CellOracle
            try:
                # Query CellOracle for genes regulated by this TF
                celloracle_dir = Path('/Users/seankiewiet/dev/projects/bio/CellOracle')
                tf_query_path = celloracle_dir / 'tf_query_grn.py'

                # Get all genes regulated by this TF
                cmd = ['python', str(tf_query_path), tf_name, '--raw']
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(celloracle_dir))

                if result.returncode == 0 and result.stdout:
                    data = json.loads(result.stdout)
                    # Count unique genes regulated by this TF
                    if isinstance(data, list):
                        tf_data['targets'] = len(data)
                        tf_data['evidence_source'] = 'CellOracle GRN database'
                    else:
                        tf_data['targets'] = 0
                        tf_data['evidence_source'] = 'CellOracle (no targets found)'
                else:
                    # If query fails, mark as unknown
                    tf_data['targets'] = 0
                    tf_data['evidence_source'] = 'CellOracle query failed'

            except Exception as e:
                tf_data['targets'] = 0
                tf_data['evidence_source'] = f'Error: {str(e)[:50]}'

        # Determine targeting strategy based on topology
        print(f"\n  Network topology analysis:")
        if topology['gene_in_degree'] == 1:
            primary_tf = self.evidence['direct_tfs'][0]
            tf_targets = primary_tf.get('targets', 0)
            tf_evidence = primary_tf.get('evidence_source', 'unknown')

            print(f"    {primary_tf['tf_name']} → {tf_targets} downstream genes [Source: {tf_evidence}]")
            print(f"    {self.gene} ← {topology['gene_in_degree']} upstream TFs")

            if tf_targets > 20:
                topology['targeting_recommendation'] = 'gene_enhancer'
                topology['rationale'] = (
                    f"{primary_tf['tf_name']} regulates {tf_targets} genes (high collateral damage). "
                    f"Targeting {self.gene} enhancer is more specific."
                )
                print(f"    Decision: Target {self.gene} enhancer (more specific)")
            else:
                topology['targeting_recommendation'] = 'tf_enhancer'
                topology['rationale'] = (
                    f"{primary_tf['tf_name']} only regulates {tf_targets} genes. "
                    f"Targeting TF might be more efficient."
                )
                print(f"    Decision: Target {primary_tf['tf_name']} binding site (more efficient)")
        else:
            topology['targeting_recommendation'] = 'gene_enhancer'
            topology['rationale'] = (
                f"{self.gene} has {topology['gene_in_degree']} regulators. "
                f"Direct targeting provides best specificity."
            )

        self.evidence['network_topology'] = topology
        print(f"  ✓ Network analysis: {topology['targeting_recommendation']}")
        print(f"  ✓ Rationale: {topology['rationale']}")

    def _generate_targeting_strategy(self):
        """Generate evidence-based targeting strategy"""
        print(f"\n[6/8] Generating targeting strategy...")

        # Determine strategy based on chromatin and expression
        if self.strategy == 'auto':
            # Auto-detect based on chromatin state
            chromatin = self.evidence.get('chromatin_state', {})
            if chromatin.get('healthy') == 'closed' and chromatin.get('pathologic') == 'open':
                self.strategy = 'inhibit'
                strategy_rationale = "Prevent pathologic chromatin opening"
            elif chromatin.get('healthy') == 'open' and chromatin.get('pathologic') == 'closed':
                self.strategy = 'activate'
                strategy_rationale = "Restore healthy chromatin state"
            else:
                self.strategy = 'inhibit'  # Default
                strategy_rationale = "Default inhibition strategy"
        else:
            strategy_rationale = f"User-specified {self.strategy} strategy"

        # Find optimal coordinates based on ISM scores
        best_region = None
        best_score = 0
        all_targets = []  # Store all potential targets

        # Check PROMOTER region
        if 'promoter' in self.evidence.get('ism_scores', {}):
            ism = self.evidence['ism_scores']['promoter']
            score = ism['max_impact'] * 0.7 + ism['mean_impact'] * 0.3
            target = {
                'type': 'promoter',
                'coordinates': f"{ism['chromosome']}:{ism['start']}-{ism['end']}",
                'chromosome': ism['chromosome'],
                'start': ism['start'],
                'end': ism['end'],
                'score': score,
                'ism_max': ism['max_impact'],
                'ism_mean': ism['mean_impact']
            }
            all_targets.append(target)
            if score > best_score:
                best_score = score
                best_region = target
            print(f"  • Promoter score: {score:.2f} (max: {ism['max_impact']:.2f})")

        # Check ENHANCER regions
        for i in range(10):  # Support up to 10 enhancers
            key = f'enhancer_{i}'
            if key in self.evidence.get('ism_scores', {}):
                ism = self.evidence['ism_scores'][key]
                score = ism['max_impact'] * 0.7 + ism['mean_impact'] * 0.3
                enhancer_info = self.evidence['regulatory_regions']['enhancers'][i] if i < len(self.evidence.get('regulatory_regions', {}).get('enhancers', [])) else {}
                target = {
                    'type': enhancer_info.get('type', f'enhancer_{i}'),
                    'coordinates': f"{ism['chromosome']}:{ism['start']}-{ism['end']}",
                    'chromosome': ism['chromosome'],
                    'start': ism['start'],
                    'end': ism['end'],
                    'score': score,
                    'ism_max': ism['max_impact'],
                    'ism_mean': ism['mean_impact']
                }
                all_targets.append(target)
                if score > best_score:
                    best_score = score
                    best_region = target
                print(f"  • Enhancer {i+1} score: {score:.2f} (max: {ism['max_impact']:.2f})")

        # Check TF BINDING sites
        for tf_data in self.evidence['direct_tfs']:
            tf_name = tf_data['tf_name']
            key = f"{tf_name}_binding"
            if key in self.evidence.get('ism_scores', {}):
                ism = self.evidence['ism_scores'][key]
                score = ism['max_impact'] * 0.7 + ism['mean_impact'] * 0.3
                target = {
                    'type': f'TF_{tf_name}_binding',
                    'coordinates': f"{ism['chromosome']}:{ism['start']}-{ism['end']}",
                    'chromosome': ism['chromosome'],
                    'start': ism['start'],
                    'end': ism['end'],
                    'score': score,
                    'ism_max': ism['max_impact'],
                    'ism_mean': ism['mean_impact']
                }
                all_targets.append(target)
                if score > best_score:
                    best_score = score
                    best_region = target
                print(f"  • {tf_name} binding site score: {score:.2f} (max: {ism['max_impact']:.2f})")
            elif tf_data.get('ism_score', 0) > 0:
                # Legacy format
                print(f"  • {tf_name} binding site score: {tf_data['ism_score']:.2f}")

        # For backward compatibility with gene_enhancer
        if 'gene_enhancer' in self.evidence.get('ism_scores', {}):
            ism = self.evidence['ism_scores']['gene_enhancer']
            score = ism['max_impact'] * 0.7 + ism['mean_impact'] * 0.3
            if score > best_score:
                best_score = score
                best_region = {
                    'type': 'gene_enhancer',
                    'coordinates': f"{ism['chromosome']}:{ism['start']}-{ism['end']}",
                    'chromosome': ism['chromosome'],
                    'start': ism['start'],
                    'end': ism['end'],
                    'score': score,
                    'ism_max': ism['max_impact'],
                    'ism_mean': ism['mean_impact']
                }

        # Store all targets for reporting
        self.evidence['all_targets'] = sorted(all_targets, key=lambda x: x['score'], reverse=True)

        if best_region:
            self.evidence['optimal_coordinates'] = best_region
            self.evidence['targeting_evidence'] = {
                'strategy': self.strategy,
                'rationale': strategy_rationale,
                'target_type': best_region['type'],
                'coordinates': best_region['coordinates'],
                'expected_impact': best_score,
                'confidence': 'high' if best_score > 1000 else 'medium'
            }

            print(f"  ✓ Strategy: {self.strategy}")
            print(f"  ✓ Target: {best_region['coordinates']}")
            print(f"  ✓ Expected impact: {best_score:.2f}")

    def _calculate_confidence_score(self):
        """Calculate overall confidence in the targeting recommendation"""
        print(f"\n[7/8] Calculating confidence score...")

        scores = {}

        # ISM impact score (0-1)
        if 'gene_enhancer' in self.evidence.get('ism_scores', {}):
            ism = self.evidence['ism_scores']['gene_enhancer']
            # Normalize to 0-1 (assuming max impact of 100000)
            scores['ism_impact'] = min(ism['max_impact'] / 100000, 1.0)
        else:
            scores['ism_impact'] = 0.0

        # Network specificity score (0-1)
        topology = self.evidence.get('network_topology', {})
        if topology.get('gene_in_degree', 0) == 1:
            scores['network_specificity'] = 0.9  # Single regulator is good
        elif topology.get('gene_in_degree', 0) <= 3:
            scores['network_specificity'] = 0.7
        else:
            scores['network_specificity'] = 0.5

        # Chromatin accessibility score (0-1)
        chromatin = self.evidence.get('chromatin_state', {})
        if chromatin.get('healthy') and chromatin.get('pathologic'):
            if chromatin['healthy'] != chromatin['pathologic']:
                scores['chromatin_accessibility'] = 0.9  # Clear difference
            else:
                scores['chromatin_accessibility'] = 0.5
        else:
            scores['chromatin_accessibility'] = 0.3

        # TF binding score (0-1)
        if self.evidence.get('direct_tfs'):
            scores['tf_binding'] = min(len(self.evidence['direct_tfs']) / 10, 1.0)
        else:
            scores['tf_binding'] = 0.0

        # Distance to TSS score (0-1)
        if self.evidence.get('optimal_coordinates'):
            coord = self.evidence['optimal_coordinates']
            tss = self.evidence['gene_info'].get('tss', 0)
            if tss:
                distance = abs((coord['start'] + coord['end']) / 2 - tss)
                # Closer is better, max distance 10kb
                scores['distance_to_tss'] = max(0, 1 - distance / 10000)
            else:
                scores['distance_to_tss'] = 0.5
        else:
            scores['distance_to_tss'] = 0.0

        # Calculate weighted confidence
        confidence = 0.0
        for metric, weight in self.scoring_weights.items():
            confidence += scores.get(metric, 0.0) * weight

        self.evidence['confidence_score'] = round(confidence, 3)
        self.evidence['confidence_components'] = scores

        print(f"  ✓ Overall confidence: {confidence:.1%}")
        print(f"  ✓ ISM impact: {scores['ism_impact']:.2f}")
        print(f"  ✓ Network specificity: {scores['network_specificity']:.2f}")

    def _predict_chromatin_state(self, chrom: str, start: int, end: int,
                                 condition: str = 'healthy') -> Optional[Dict]:
        """Predict chromatin state using AlphaGenome"""
        try:
            from AlphaGenome import AlphaGenome

            ag = AlphaGenome()

            # Set ontology based on condition
            if condition == 'healthy':
                ontology = self.healthy_ontology
            else:
                ontology = self.pathologic_ontology

            # Adjust window to valid size
            window_size = end - start
            if window_size < 16384:
                center = (start + end) // 2
                start = center - 8192
                end = center + 8192

            # Ensure chromosome format
            if not chrom.startswith('chr'):
                chrom = f'chr{chrom}'

            result = ag.predict_interval(
                chromosome=chrom,
                start=int(start),
                end=int(end),
                ontology=ontology,
                organism=self.organism,
                run_name=f"{self.gene}_{condition}_chromatin",
                output_dir=str(self.results_dir)
            )

            if result and 'output_file' in result:
                data = np.load(result['output_file'])

                # Analyze DNase signal
                if 'dnase' in data:
                    dnase = data['dnase']
                    if dnase.ndim > 1:
                        dnase = dnase.flatten()
                    signal = float(np.mean(dnase))

                    return {
                        'state': 'open' if signal > 0.5 else 'closed',
                        'signal': signal,
                        'condition': condition
                    }

        except Exception as e:
            print(f"    ⚠ Chromatin prediction failed: {e}")

        return None

    def _save_evidence_report(self):
        """Save comprehensive evidence report"""
        print(f"\n[8/8] Saving evidence report...")

        # Save JSON evidence
        output_file = self.results_dir / f"{self.gene}_evidence.json"
        with open(output_file, 'w') as f:
            json.dump(self.evidence, f, indent=2, default=str)

        print(f"  ✓ Evidence saved to {output_file}")

        # Generate human-readable report
        report = self._generate_report()
        report_file = self.results_dir / f"{self.gene}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"  ✓ Report saved to {report_file}")

        # Generate TSV for bulk scoring
        self._save_bulk_scoring_tsv()

    def _generate_report(self) -> str:
        """Generate human-readable evidence report"""
        report = f"""
{'='*80}
CRISPR TARGET EVIDENCE REPORT: {self.gene}
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

GENE INFORMATION
----------------
"""
        if self.evidence['gene_info'].get('symbol'):
            info = self.evidence['gene_info']
            report += f"Symbol: {info['symbol']}\n"
            report += f"Location: {info.get('location', 'N/A')}\n"
            report += f"TSS: {info.get('tss', 'N/A'):,}\n"
            report += f"Biotype: {info.get('biotype', 'N/A')}\n"

        report += f"\n\nOPTIMAL CRISPR TARGET\n"
        report += f"---------------------\n"

        if self.evidence.get('optimal_coordinates'):
            coord = self.evidence['optimal_coordinates']
            report += f"Coordinates: {coord['coordinates']}\n"
            report += f"Type: {coord['type'].upper().replace('_', ' ')}\n"
            report += f"ISM Impact: {coord['ism_max']:.2f}\n"
            report += f"Score: {coord['score']:.2f}\n"

        # Report all candidate targets
        if self.evidence.get('all_targets'):
            report += f"\n\nALL CANDIDATE TARGETS (Ranked by Score)\n"
            report += f"-----------------------------------------\n"
            for i, target in enumerate(self.evidence['all_targets'], 1):
                target_type = target['type'].upper().replace('_', ' ')
                report += f"\n{i}. {target_type}\n"
                report += f"   Coordinates: {target['coordinates']}\n"
                report += f"   ISM Score: {target['score']:.2f}\n"
                report += f"   Max Impact: {target['ism_max']:.2f}\n"
                report += f"   Mean Impact: {target['ism_mean']:.2f}\n"

        report += f"\n\nTARGETING STRATEGY\n"
        report += f"------------------\n"

        if self.evidence.get('targeting_evidence'):
            target = self.evidence['targeting_evidence']
            report += f"Strategy: {target['strategy']}\n"
            report += f"Rationale: {target['rationale']}\n"
            report += f"Confidence: {target['confidence']}\n"

        report += f"\n\nCONFIDENCE ASSESSMENT\n"
        report += f"---------------------\n"
        report += f"Overall Score: {self.evidence.get('confidence_score', 0):.1%}\n"

        if self.evidence.get('confidence_components'):
            comp = self.evidence['confidence_components']
            report += f"  ISM Impact: {comp.get('ism_impact', 0):.2f} (AlphaGenome)\n"
            report += f"  Network Specificity: {comp.get('network_specificity', 0):.2f} (CellOracle)\n"
            report += f"  Chromatin Accessibility: {comp.get('chromatin_accessibility', 0):.2f} (AlphaGenome)\n"

        report += f"\n\nNETWORK TOPOLOGY\n"
        report += f"----------------\n"

        if self.evidence.get('network_topology'):
            topo = self.evidence['network_topology']
            report += f"Gene Regulators: {topo.get('gene_in_degree', 0)} (Source: CellOracle)\n"
            report += f"Recommendation: {topo.get('targeting_recommendation', 'N/A')}\n"
            report += f"Rationale: {topo.get('rationale', 'N/A')}\n"

        # Add TF details if available
        if self.evidence.get('direct_tfs'):
            report += f"\nTF REGULATORY EVIDENCE\n"
            report += f"----------------------\n"
            for tf in self.evidence['direct_tfs'][:3]:
                report += f"  {tf['tf_name']}:\n"
                report += f"    - Downstream targets: {tf.get('targets', 'unknown')}\n"
                report += f"    - Evidence source: {tf.get('evidence_source', 'unknown')}\n"
                report += f"    - ISM score: {tf.get('ism_score', 0):.2f}\n"

        report += f"\n{'='*80}\n"

        return report

    def _save_bulk_scoring_tsv(self):
        """Save comprehensive TSV with all AlphaGenome data for bulk scoring"""
        tsv_file = self.results_dir / f"{self.gene}_targets.tsv"

        with open(tsv_file, 'w') as f:
            # Write header row
            f.write("gene\ttarget_type\tmode\tchromosome\tposition\tstart\tend\t")
            f.write("ism_max\tism_mean\tdistance_to_tss\tchromatin_state\tdnase_signal\t")
            f.write("accessibility\tcell_type\tregion_type\ttf_name\ttf_targets\t")
            f.write("confidence_score\tnotes\n")

            # Get chromatin state for the gene
            chromatin = self.evidence.get('chromatin_state', {})
            chrom_state = chromatin.get('state', 'unknown')
            dnase = chromatin.get('dnase_signal', 0)
            accessibility = chromatin.get('accessibility', 'unknown')
            cell_type = chromatin.get('cell_type', 'CL:0000127')

            # Determine mode based on strategy
            mode = self.strategy if self.strategy != 'auto' else 'inhibit'

            # 1. Write GENE PROMOTER row
            if 'promoter' in self.evidence.get('ism_scores', {}):
                promoter = self.evidence['ism_scores']['promoter']
                position = (promoter['start'] + promoter['end']) // 2
                distance = abs(position - self.evidence['gene_info'].get('tss', 0))

                f.write(f"{self.gene}\tgene_promoter\t{mode}\t")
                f.write(f"{promoter['chromosome']}\t{position}\t{promoter['start']}\t{promoter['end']}\t")
                f.write(f"{promoter['max_impact']:.2f}\t{promoter['mean_impact']:.2f}\t{distance}\t")
                f.write(f"{chrom_state}\t{dnase:.4f}\t{accessibility}\t{cell_type}\t")
                f.write(f"promoter\t-\t-\t")  # No TF for gene row
                f.write(f"{self.evidence.get('confidence_score', 0):.3f}\t")
                f.write(f"{'FAVORED' if dnase > 0.3 else 'CLOSED_CHROMATIN'}\n")

            # 2. Write GENE ENHANCER rows if found
            for i in range(10):
                key = f'enhancer_{i}'
                if key in self.evidence.get('ism_scores', {}):
                    enhancer = self.evidence['ism_scores'][key]
                    position = (enhancer['start'] + enhancer['end']) // 2
                    distance = abs(position - self.evidence['gene_info'].get('tss', 0))

                    # Get enhancer-specific chromatin if available
                    enhancer_info = {}
                    if i < len(self.evidence.get('regulatory_regions', {}).get('enhancers', [])):
                        enhancer_info = self.evidence['regulatory_regions']['enhancers'][i]
                    enhancer_dnase = enhancer_info.get('healthy_dnase', dnase)
                    enhancer_state = 'open' if enhancer_dnase > 0.3 else 'closed'

                    f.write(f"{self.gene}\tgene_enhancer_{i}\t{mode}\t")
                    f.write(f"{enhancer['chromosome']}\t{position}\t{enhancer['start']}\t{enhancer['end']}\t")
                    f.write(f"{enhancer['max_impact']:.2f}\t{enhancer['mean_impact']:.2f}\t{distance}\t")
                    f.write(f"{enhancer_state}\t{enhancer_dnase:.4f}\t")
                    f.write(f"{'open' if enhancer_dnase > 0.3 else 'closed'}\t{cell_type}\t")
                    f.write(f"{enhancer_info.get('type', f'enhancer_{i}')}\t-\t-\t")
                    f.write(f"{self.evidence.get('confidence_score', 0):.3f}\t")
                    f.write(f"{'FAVORED' if enhancer_dnase > 0.3 else 'CLOSED_CHROMATIN'}\n")

            # 3. Write TF BINDING SITE rows
            for tf in self.evidence.get('direct_tfs', []):
                tf_name = tf['tf_name']
                tf_targets = tf.get('targets', 0)

                # Write a row for this TF if it has ISM score
                if tf.get('ism_score', 0) > 0:
                    # Use promoter region as default location
                    if 'promoter' in self.evidence.get('ism_scores', {}):
                        promoter = self.evidence['ism_scores']['promoter']
                        # TF binding typically upstream of promoter
                        tf_position = promoter['start'] - 100

                        f.write(f"{self.gene}\ttf_binding\t{mode}\t")
                        f.write(f"{promoter['chromosome']}\t{tf_position}\t")
                        f.write(f"{tf_position - 25}\t{tf_position + 25}\t")
                        f.write(f"{tf.get('ism_score', 0):.2f}\t0.00\t")
                        f.write(f"{abs(tf_position - self.evidence['gene_info'].get('tss', 0))}\t")
                        f.write(f"{chrom_state}\t{dnase:.4f}\t{accessibility}\t{cell_type}\t")
                        f.write(f"tf_binding\t{tf_name}\t{tf_targets}\t")
                        f.write(f"{self.evidence.get('confidence_score', 0):.3f}\t")
                        f.write(f"{'FAVORED' if dnase > 0.3 else 'CLOSED_CHROMATIN'}\n")

        print(f"  ✓ Comprehensive bulk scoring TSV saved to {tsv_file}")
        if dnase < 0.3:
            print(f"    ⚠ NOTE: Closed chromatin detected (DNase={dnase:.3f}) - marked in TSV")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Evidence-Based CRISPR Target Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  %(prog)s --gene IL1B

  # Activation strategy with deep ISM
  %(prog)s --gene GFAP --strategy activate --ism-depth 2

  # Inhibition with custom ISM window
  %(prog)s --gene STAT3 --strategy inhibit --ism-bps 200

  # Full analysis with all evidence
  %(prog)s --gene ARG2 --depth 3 --ism-depth 2
""")

    parser.add_argument('--gene', required=True,
                       help='Gene symbol to analyze')
    parser.add_argument('--organism', default='human',
                       choices=['human', 'mouse'],
                       help='Organism (default: human)')
    parser.add_argument('--depth', type=int, default=3,
                       help='Maximum cascade depth (default: 3)')
    parser.add_argument('--ism-bps', type=int, default=100,
                       help='ISM region size in bp (default: 100)')
    parser.add_argument('--ism-depth', type=int, default=2, choices=[1, 2],
                       help='ISM depth: 1=gene only, 2=gene+TFs (default: 2)')
    parser.add_argument('--strategy', default='auto',
                       choices=['inhibit', 'activate', 'auto'],
                       help='Targeting strategy (default: auto-detect)')
    parser.add_argument('--healthy-ontology',
                       help='Cell type ontology for healthy state (default: CL:0000127 for astrocyte)')
    parser.add_argument('--pathologic-ontology',
                       help='Cell type ontology for pathologic state (default: same as healthy)')
    parser.add_argument('--output',
                       help='Output directory (default: results/GENE_analysis/)')
    parser.add_argument('--raw', action='store_true',
                       help='Output raw JSON to stdout for pipeline integration')

    args = parser.parse_args()

    # Run analysis
    analyzer = CRISPRCascadeAnalyzer(
        gene=args.gene,
        organism=args.organism,
        max_depth=args.depth,
        ism_bps=args.ism_bps,
        ism_depth=args.ism_depth,
        strategy=args.strategy,
        healthy_ontology=args.healthy_ontology,
        pathologic_ontology=args.pathologic_ontology
    )

    # Set custom output directory if provided
    if args.output:
        analyzer.results_dir = Path(args.output)
        analyzer.results_dir.mkdir(parents=True, exist_ok=True)

    evidence = analyzer.analyze()

    # Output raw JSON if requested (for pipeline integration)
    if args.raw:
        # Output minimal JSON for bulk processing
        raw_output = {
            'gene': args.gene,
            'optimal_target': evidence.get('optimal_coordinates', {}),
            'all_targets': evidence.get('all_targets', []),
            'confidence': evidence.get('confidence_score', 0),
            'strategy': evidence.get('targeting_evidence', {}).get('strategy', 'unknown'),
            'tfs': [tf['tf_name'] for tf in evidence.get('direct_tfs', [])],
            'chromatin': {
                'healthy': evidence.get('chromatin_state', {}).get('healthy', 'unknown'),
                'pathologic': evidence.get('chromatin_state', {}).get('pathologic', 'unknown')
            }
        }
        print(json.dumps(raw_output, indent=2))
        return

    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

    if evidence.get('optimal_coordinates'):
        coord = evidence['optimal_coordinates']
        print(f"✅ Optimal Target: {coord['coordinates']}")
        print(f"✅ ISM Impact: {coord['ism_max']:.2f}")
        print(f"✅ Confidence: {evidence.get('confidence_score', 0):.1%}")
        print(f"✅ Strategy: {evidence.get('targeting_evidence', {}).get('strategy', 'N/A')}")
    else:
        print("⚠ No optimal target identified")

    print(f"\n📁 Results saved to: results/{args.gene}_analysis/")


if __name__ == '__main__':
    main()