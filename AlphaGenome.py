#!/usr/bin/env python3
"""
AlphaGenome API Interface - Optimized Version
Unified interface for AlphaGenome regulatory prediction and scoring

-----------------------------------------------------------------------
Copyright (c) 2025 Sean Kiewiet. All rights reserved.
-----------------------------------------------------------------------
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from functools import lru_cache
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Import AlphaGenome modules once at module level for efficiency
try:
    from alphagenome.models import dna_client
    from alphagenome.data import genome
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False
    logger.warning("AlphaGenome package not installed")

# Constants
VALID_INTERVAL_SIZES = [16384, 131072, 524288, 1048576]
VALID_SEQUENCE_SIZES = [16384, 131072, 524288, 1048576]
VALID_CONTEXT_SIZES = [2048, 16384, 131072, 524288, 1048576]
DEFAULT_ONTOLOGY = 'CL:0000127'  # Astrocyte
DEFAULT_CONTEXT_SIZE = 131072

class AlphaGenome:
    """Optimized AlphaGenome API Interface"""

    # Class-level caches to avoid repeated initialization
    _model = None
    _ontology_db = None
    _api_key_loaded = False

    def __init__(self, update_ontologies=False, verbose=False):
        """Initialize AlphaGenome interface with caching"""
        if not ALPHAGENOME_AVAILABLE:
            print("❌ AlphaGenome package not installed")
            print("Install with: pip install alphagenome")
            sys.exit(1)

        self.verbose = verbose
        self.model = self._get_or_create_model()

        # Only load ontologies if needed
        if update_ontologies:
            self._update_ontology_database()

    @classmethod
    def _get_or_create_model(cls):
        """Get cached model or create new one (singleton pattern)"""
        if cls._model is None:
            api_key = cls._load_api_key()
            if api_key:
                cls._model = dna_client.create(api_key)
                print("✅ AlphaGenome API Key Located")
            else:
                print("❌ No API key found")
                print("Visit https://alphagenome.com to get your API key")
                sys.exit(1)
        return cls._model

    @classmethod
    @lru_cache(maxsize=1)
    def _load_api_key(cls):
        """Load API key once and cache it"""
        config_path = Path.home() / '.alphagenome' / 'config.json'

        if not config_path.exists():
            # Check environment variable as fallback
            api_key = os.environ.get('ALPHAGENOME_API_KEY')
            if api_key:
                return api_key
            return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('api_key')
        except Exception as e:
            logger.error(f"Error loading API key: {e}")
            return None

    def get_available_organisms(self):
        """Get list of available organisms from API"""
        try:
            # These are the supported organisms in AlphaGenome
            organisms = {
                'human': 'Homo sapiens',
                'mouse': 'Mus musculus'
            }
            return organisms
        except:
            return {'human': 'Homo sapiens', 'mouse': 'Mus musculus'}

    def get_available_cell_types(self):
        """Get all available cell type ontologies from AlphaGenome API"""
        try:
            # Query the model's output metadata for available cell types
            metadata = self.model.output_metadata()

            # Get RNA_SEQ tracks dataframe which has the most comprehensive list
            rna_df = metadata.rna_seq

            # Extract unique cell type ontologies with their descriptions
            cell_types = {}

            for _, row in rna_df.iterrows():
                curie = row['ontology_curie']
                biosample = row['biosample_name']

                # Filter for cell type ontologies (CL:)
                if curie and curie.startswith('CL:'):
                    if curie not in cell_types:
                        cell_types[curie] = biosample

            # Return sorted dictionary
            return dict(sorted(cell_types.items()))

        except Exception as e:
            # Fallback to a minimal set if API fails
            print(f"Warning: Could not fetch cell types from API: {e}")
            return {
                'CL:0000127': 'astrocyte',
                'CL:0000540': 'neuron',
                'CL:0000084': 'T-cell'
            }

    def get_all_available_ontologies(self):
        """Query AlphaGenome API for all available ontologies"""
        try:
            # Get the model's output metadata
            metadata = self.model.output_metadata()

            # Combine ontologies from all output types for completeness
            all_ontologies = {}

            # Check each output type's dataframe
            for output_type in ['rna_seq', 'atac', 'dnase', 'chip_tf', 'chip_histone']:
                if hasattr(metadata, output_type):
                    df = getattr(metadata, output_type)
                    if 'ontology_curie' in df.columns:
                        for _, row in df.iterrows():
                            curie = row['ontology_curie']
                            biosample = row.get('biosample_name', curie)

                            if curie and curie not in all_ontologies:
                                all_ontologies[curie] = biosample

            return dict(sorted(all_ontologies.items()))

        except Exception as e:
            print(f"Warning: Could not fetch ontologies from API: {e}")
            return {}

    def get_available_tissues(self):
        """Get all available tissue types from AlphaGenome API"""
        try:
            # Query all ontologies and filter for tissues
            all_ontologies = self.get_all_available_ontologies()

            # Filter for UBERON (tissue) ontologies
            tissues = {k: v for k, v in all_ontologies.items() if k.startswith('UBERON:')}

            if tissues:
                return dict(sorted(tissues.items()))

        except Exception as e:
            print(f"Warning: Could not fetch tissues from API: {e}")

        # Minimal fallback
        return {
            'UBERON:0000955': 'brain',
            'UBERON:0002107': 'liver',
            'UBERON:0002048': 'lung'
        }

    @staticmethod
    @lru_cache(maxsize=2)
    def _get_organism_enum(organism: str):
        """Convert organism string to enum (cached)"""
        if organism.lower() == 'human':
            return dna_client.Organism.HOMO_SAPIENS
        else:
            return dna_client.Organism.MUS_MUSCULUS

    @staticmethod
    def _validate_interval_size(size: int, valid_sizes: list) -> bool:
        """Validate interval size"""
        if size not in valid_sizes:
            print(f"❌ Invalid size {size}. Must be one of: {valid_sizes}")
            return False
        return True

    def _save_results(self, save_dict, metadata, base_name, output_dir=None):
        """Common method to save results"""
        # Set output directory
        output_path = Path(output_dir) if output_dir else Path('results')
        output_path.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays if present
        npz_file = None
        if save_dict:
            npz_file = output_path / f"{base_name}.npz"
            np.savez_compressed(npz_file, **save_dict)
            metadata['output_file'] = str(npz_file)

        # Save metadata
        json_file = output_path / f"{base_name}.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return npz_file, json_file


    # === PREDICTION METHODS ===

    def predict_interval(self, chromosome: str, start: int, end: int,
                        ontology: str = DEFAULT_ONTOLOGY, organism: str = 'human',
                        run_name: str = None, output_dir: str = None, gene_info: dict = None):
        """Predict regulatory tracks for a genomic interval"""

        # Validate size once
        interval_size = end - start
        if not self._validate_interval_size(interval_size, VALID_INTERVAL_SIZES):
            return None

        try:
            # Create interval
            interval = genome.Interval(chromosome, start, end)
            organism_enum = self._get_organism_enum(organism)

            # Make API call
            outputs = self.model.predict_interval(
                interval=interval,
                organism=organism_enum,
                ontology_terms=[ontology],
                requested_outputs=[
                    dna_client.OutputType.DNASE,
                    dna_client.OutputType.RNA_SEQ,
                    dna_client.OutputType.CHIP_TF,
                    dna_client.OutputType.CHIP_HISTONE,
                    dna_client.OutputType.CAGE,
                    dna_client.OutputType.PROCAP
                ]
            )

            # Process outputs efficiently
            save_dict = {}
            shapes = {}

            for attr in ['dnase', 'rna_seq', 'chip_tf', 'chip_histone', 'cage', 'procap']:
                output = getattr(outputs, attr, None)
                if output is not None:
                    save_dict[attr] = output.values
                    shapes[attr] = output.values.shape

            # Prepare metadata
            base_name = run_name or f"pred_interval_{chromosome}_{start}_{end}"
            metadata = {
                'api_method': 'predict_interval',
                'chromosome': chromosome,
                'start': start,
                'end': end,
                'size': interval_size,
                'ontology': ontology,
                'organism': organism,
                'shapes': shapes
            }

            # Add gene info if provided
            if gene_info:
                metadata['gene_info'] = gene_info
                metadata['gene'] = base_name.replace('_predict', '')  # Extract gene name from run_name

            # Save results
            self._save_results(save_dict, metadata, base_name, output_dir)

            print(f"✅ Predictions saved to {base_name}.npz")
            print(f"\nVisualize with:")
            print(f"  python visualizer.py {base_name}.npz --tracks all --overlays tss gene promoter enhancer --peaks")
            return metadata

        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None

    def predict_sequence(self, sequence: str, ontology: str = DEFAULT_ONTOLOGY,
                        organism: str = 'human', run_name: str = None,
                        output_dir: str = None):
        """Predict regulatory tracks from DNA sequence"""

        # Handle file input
        if Path(sequence).exists():
            sequence = self._load_sequence_from_file(sequence)
            if not sequence:
                return None

        # Validate sequence
        sequence = sequence.upper().replace('\n', '').replace(' ', '')
        if not all(c in 'ATCG' for c in sequence):
            print("❌ Invalid sequence. Only ATCG allowed.")
            return None

        seq_len = len(sequence)
        if not self._validate_interval_size(seq_len, VALID_SEQUENCE_SIZES):
            return None

        try:
            organism_enum = self._get_organism_enum(organism)

            # Make API call
            outputs = self.model.predict_sequence(
                sequence=sequence,
                organism=organism_enum,
                ontology_terms=[ontology],
                requested_outputs=[
                    dna_client.OutputType.DNASE,
                    dna_client.OutputType.RNA_SEQ,
                    dna_client.OutputType.CHIP_TF,
                    dna_client.OutputType.CHIP_HISTONE,
                    dna_client.OutputType.CAGE,
                    dna_client.OutputType.PROCAP
                ]
            )

            # Process outputs (same as predict_interval)
            save_dict = {}
            shapes = {}

            for attr in ['dnase', 'rna_seq', 'chip_tf', 'chip_histone', 'cage', 'procap']:
                output = getattr(outputs, attr, None)
                if output is not None:
                    save_dict[attr] = output.values
                    shapes[attr] = output.values.shape

            # Save results
            base_name = run_name or f"pred_sequence_{seq_len}bp"
            metadata = {
                'api_method': 'predict_sequence',
                'sequence_length': seq_len,
                'sequence_first_50bp': sequence[:50],
                'sequence_last_50bp': sequence[-50:],
                'ontology': ontology,
                'organism': organism,
                'shapes': shapes
            }

            self._save_results(save_dict, metadata, base_name, output_dir)
            print(f"✅ Predictions saved to {base_name}.npz")
            print(f"\nVisualize with:")
            print(f"  python visualizer.py {base_name}.npz --tracks all --overlays tss gene promoter enhancer --peaks")
            return metadata

        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None

    def predict_variant(self, chromosome: str, position: int, ref: str, alt: str,
                       context_start: int = None, context_end: int = None,
                       ontology: str = DEFAULT_ONTOLOGY, organism: str = 'human',
                       run_name: str = None, output_dir: str = None):
        """Predict variant effect on regulatory tracks"""

        # Calculate context if not provided
        if context_start is None or context_end is None:
            context_start = position - DEFAULT_CONTEXT_SIZE // 2
            context_end = position + DEFAULT_CONTEXT_SIZE // 2

        context_size = context_end - context_start
        if not self._validate_interval_size(context_size, VALID_CONTEXT_SIZES):
            return None

        try:
            # Create interval and variant
            interval = genome.Interval(chromosome, context_start, context_end)
            variant = genome.Variant(chromosome, position, ref, alt)
            organism_enum = self._get_organism_enum(organism)

            # Make API call
            outputs = self.model.predict_variant(
                interval=interval,
                variant=variant,
                organism=organism_enum,
                ontology_terms=[ontology],
                requested_outputs=[
                    dna_client.OutputType.DNASE,
                    dna_client.OutputType.RNA_SEQ,
                    dna_client.OutputType.CHIP_TF,
                    dna_client.OutputType.CHIP_HISTONE,
                    dna_client.OutputType.CAGE,
                    dna_client.OutputType.PROCAP
                ]
            )

            # Process ref and alt outputs
            save_dict = {}
            shapes = {}

            for allele_type in ['reference', 'alternate']:
                allele_output = getattr(outputs, allele_type, None)
                if allele_output:
                    suffix = '_ref' if allele_type == 'reference' else '_alt'
                    for attr in ['dnase', 'rna_seq', 'chip_tf', 'chip_histone', 'cage', 'procap']:
                        output = getattr(allele_output, attr, None)
                        if output is not None:
                            key = f"{attr}{suffix}"
                            save_dict[key] = output.values
                            shapes[key] = output.values.shape

            # Save results
            base_name = run_name or f"pred_variant_{chromosome}_{position}_{ref}_{alt}"
            metadata = {
                'api_method': 'predict_variant',
                'chromosome': chromosome,
                'position': position,
                'ref': ref,
                'alt': alt,
                'variant': f"{chromosome}:{position} {ref}>{alt}",
                'context_start': context_start,
                'context_end': context_end,
                'context_size': context_size,
                'ontology': ontology,
                'organism': organism,
                'shapes': shapes
            }

            self._save_results(save_dict, metadata, base_name, output_dir)
            print(f"✅ Predictions saved to {base_name}.npz")
            print(f"\nVisualize with:")
            print(f"  python visualizer.py {base_name}.npz --tracks all --overlays tss gene promoter enhancer --peaks")
            return metadata

        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None

    # === SCORING METHODS ===

    def score_variant(self, chromosome: str, position: int, ref: str, alt: str,
                     context_start: int = None, context_end: int = None,
                     organism: str = 'human', run_name: str = None,
                     output_dir: str = None):
        """Score variant impact using multiple scoring methods"""

        # Calculate context if not provided
        if context_start is None or context_end is None:
            context_start = position - DEFAULT_CONTEXT_SIZE // 2
            context_end = position + DEFAULT_CONTEXT_SIZE // 2

        context_size = context_end - context_start
        if not self._validate_interval_size(context_size, VALID_CONTEXT_SIZES):
            return None

        try:
            # Create interval and variant
            interval = genome.Interval(chromosome, context_start, context_end)
            variant = genome.Variant(chromosome, position, ref, alt)
            organism_enum = self._get_organism_enum(organism)

            # Make API call
            scores = self.model.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=[],  # Use default scorers
                organism=organism_enum
            )

            # Process scores
            save_dict = {}
            score_results = []

            for idx, score_data in enumerate(scores):
                scorer_info = {'index': idx}

                # Get scorer metadata
                if hasattr(score_data, 'uns') and 'variant_scorer' in score_data.uns:
                    scorer_info['scorer_id'] = str(score_data.uns['variant_scorer'])

                # Get scores
                if hasattr(score_data, 'X') and score_data.X is not None:
                    scorer_info['shape'] = score_data.X.shape
                    save_dict[f'scorer_{idx}'] = score_data.X

                    # Extract single value if possible
                    if score_data.X.shape == (1, 1):
                        scorer_info['value'] = float(score_data.X[0, 0])

                score_results.append(scorer_info)

            # Save results
            base_name = run_name or f"score_variant_{chromosome}_{position}_{ref}_{alt}"
            metadata = {
                'api_method': 'score_variant',
                'chromosome': chromosome,
                'position': position,
                'ref': ref,
                'alt': alt,
                'variant': f"{chromosome}:{position} {ref}>{alt}",
                'context_start': context_start,
                'context_end': context_end,
                'organism': organism,
                'n_scorers': len(scores),
                'scores': score_results
            }

            self._save_results(save_dict, metadata, base_name, output_dir)
            print(f"✅ Scores saved to {base_name}.json")
            print(f"\nView scores with:")
            print(f"  cat {base_name}.json | python -m json.tool | grep -A5 impact")
            print(f"  python visualizer.py {base_name}.npz --tracks all --overlays tss gene promoter enhancer --peaks")
            return metadata

        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None

    def score_interval(self, chromosome: str, start: int, end: int,
                      organism: str = 'human', run_name: str = None,
                      output_dir: str = None, gene_info: dict = None):
        """Score regulatory activity across a genomic interval"""

        interval_size = end - start
        if not self._validate_interval_size(interval_size, VALID_INTERVAL_SIZES):
            return None

        try:
            interval = genome.Interval(chromosome, start, end)
            organism_enum = self._get_organism_enum(organism)

            # Make API call
            scores = self.model.score_interval(
                interval=interval,
                interval_scorers=[],  # Use default scorers
                organism=organism_enum
            )

            # Process scores (similar to score_variant)
            save_dict = {}
            score_results = []

            for idx, score_data in enumerate(scores):
                scorer_info = {'index': idx}

                if hasattr(score_data, 'uns'):
                    if 'interval_scorer' in score_data.uns:
                        scorer_info['scorer_id'] = str(score_data.uns['interval_scorer'])

                if hasattr(score_data, 'X') and score_data.X is not None:
                    scorer_info['shape'] = score_data.X.shape
                    save_dict[f'scorer_{idx}'] = score_data.X

                    # Calculate statistics
                    if score_data.X.size > 0:
                        scorer_info['values'] = {
                            'mean': float(np.mean(score_data.X)),
                            'std': float(np.std(score_data.X)),
                            'min': float(np.min(score_data.X)),
                            'max': float(np.max(score_data.X))
                        }

                score_results.append(scorer_info)

            # Save results
            base_name = run_name or f"score_interval_{chromosome}_{start}_{end}"
            metadata = {
                'api_method': 'score_interval',
                'chromosome': chromosome,
                'start': start,
                'end': end,
                'interval': f"{chromosome}:{start}-{end}",
                'interval_size': interval_size,
                'organism': organism,
                'n_scorers': len(scores),
                'scores': score_results
            }

            self._save_results(save_dict, metadata, base_name, output_dir)
            print(f"✅ Scores saved to {base_name}.json")
            print(f"\nView scores with:")
            print(f"  cat {base_name}.json | python -m json.tool | grep -A5 impact")
            print(f"  python visualizer.py {base_name}.npz --tracks all --overlays tss gene promoter enhancer --peaks")
            return metadata

        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None

    def score_ism_variants(self, chromosome: str, ism_start: int, ism_end: int,
                          context_start: int = None, context_end: int = None,
                          organism: str = 'human', run_name: str = None,
                          output_dir: str = None):
        """Score In-Silico Mutagenesis variants"""

        # Calculate context if not provided
        if context_start is None or context_end is None:
            ism_center = (ism_start + ism_end) // 2
            context_start = max(0, ism_center - DEFAULT_CONTEXT_SIZE // 2)
            context_end = ism_center + DEFAULT_CONTEXT_SIZE // 2

        context_size = context_end - context_start
        if not self._validate_interval_size(context_size, VALID_CONTEXT_SIZES):
            return None

        ism_size = ism_end - ism_start
        expected_mutations = ism_size * 3

        print(f"\nScoring ISM variants")
        print(f"ISM region: {chromosome}:{ism_start}-{ism_end} ({ism_size} bp)")
        print(f"Expected mutations: {expected_mutations}")

        try:
            context_interval = genome.Interval(chromosome, context_start, context_end)
            ism_interval = genome.Interval(chromosome, ism_start, ism_end)
            organism_enum = self._get_organism_enum(organism)

            # Make API call
            print("\nRunning ISM (this may take a while)...")
            scores_list = self.model.score_ism_variants(
                interval=context_interval,
                ism_interval=ism_interval,
                variant_scorers=[],
                organism=organism_enum,
                progress_bar=True
            )

            # Process results
            all_mutations = []
            save_dict = {}

            for mutation_idx, mutation_scores in enumerate(scores_list):
                # Get mutation info
                if mutation_scores and len(mutation_scores) > 0:
                    first_scorer = mutation_scores[0]
                    if hasattr(first_scorer, 'uns') and 'variant' in first_scorer.uns:
                        all_mutations.append({
                            'mutation_idx': mutation_idx,
                            'variant': str(first_scorer.uns['variant'])
                        })

                # Process each scorer
                for scorer_idx, score_data in enumerate(mutation_scores):
                    if hasattr(score_data, 'X') and score_data.X is not None:
                        key = f'mutation_{mutation_idx}_scorer_{scorer_idx}'
                        save_dict[key] = score_data.X

            # Save results
            base_name = run_name or f"score_ism_{chromosome}_{ism_start}_{ism_end}"
            metadata = {
                'api_method': 'score_ism_variants',
                'chromosome': chromosome,
                'ism_start': ism_start,
                'ism_end': ism_end,
                'ism_size': ism_size,
                'context_start': context_start,
                'context_end': context_end,
                'organism': organism,
                'n_mutations': len(all_mutations),
                'mutations': all_mutations[:100]  # First 100 for preview
            }

            self._save_results(save_dict, metadata, base_name, output_dir)
            print(f"✅ ISM scores saved to {base_name}.json")
            print(f"  Mutations tested: {len(all_mutations)}")
            print(f"\nVisualize ISM results:")
            print(f"  python ism_heatmap.py {base_name}.npz {base_name}.json {base_name}_heatmap.png")
            print(f"  python visualizer.py {base_name}.npz --tracks all --overlays tss gene promoter enhancer --peaks")
            return metadata

        except Exception as e:
            print(f"❌ API call failed: {e}")
            return None

    @staticmethod
    def _load_sequence_from_file(filepath):
        """Load DNA sequence from FASTA file"""
        sequence = ""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.startswith('>'):
                        sequence += line.strip()
            print(f"Loaded sequence from file: {len(sequence)} bp")
            return sequence
        except Exception as e:
            print(f"❌ Error loading sequence: {e}")
            return None

    def gene_baseline(self, gene: str, cell_type: str = DEFAULT_ONTOLOGY,
                      organism: str = 'human', window: int = 524288,
                      run_name: str = None, output_dir: str = None):
        """Generate baseline regulatory predictions for a gene

        This replaces get_gene_ground_truth.py functionality
        """
        print(f"\n{'='*60}")
        print(f"🧬 GENE REGULATORY BASELINE: {gene}")
        print('='*60)

        # Get gene coordinates from Ensembl
        print(f"\n📍 Fetching {gene} coordinates...")
        try:
            import requests
            server = "https://rest.ensembl.org"
            species = "homo_sapiens" if organism == 'human' else "mus_musculus"
            ext = f"/lookup/symbol/{species}/{gene}?expand=1"

            r = requests.get(server + ext, headers={"Content-Type": "application/json"})
            if r.ok:
                data = r.json()
                gene_info = {
                    'chromosome': f"chr{data.get('seq_region_name')}",
                    'start': data.get('start'),
                    'end': data.get('end'),
                    'strand': data.get('strand'),
                    'description': data.get('description', ''),
                    'biotype': data.get('biotype')
                }
                print(f"  Gene: {gene_info['chromosome']}:{gene_info['start']:,}-{gene_info['end']:,}")
                print(f"  Type: {gene_info['biotype']}")
                if gene_info['description']:
                    desc = gene_info['description'][:80] + '...' if len(gene_info['description']) > 80 else gene_info['description']
                    print(f"  Description: {desc}")
            else:
                print(f"❌ Could not find gene {gene}")
                return None
        except Exception as e:
            print(f"❌ Error fetching gene: {e}")
            return None

        # Calculate regulatory window centered on TSS
        # TSS is at gene start for + strand, gene end for - strand
        strand = gene_info.get('strand', 1)
        if strand > 0:  # Positive strand
            tss = gene_info['start']
        else:  # Negative strand
            tss = gene_info['end']

        half_window = window // 2
        context_start = max(1, tss - half_window)
        context_end = tss + half_window

        print(f"  Context: {gene_info['chromosome']}:{context_start:,}-{context_end:,} ({window//1024}kb)")

        # Run predictions
        print(f"\n📊 Running baseline regulatory predictions...")
        base_name = run_name or f"{gene}_baseline"

        # Predict interval
        self.predict_interval(
            chromosome=gene_info['chromosome'],
            start=context_start,
            end=context_end,
            ontology=cell_type,
            organism=organism,
            run_name=f"{base_name}_predict",
            output_dir=output_dir,
            gene_info=gene_info
        )

        # Score interval if window is large enough
        if window >= 524288:
            print(f"\n📈 Scoring regulatory activity...")
            self.score_interval(
                chromosome=gene_info['chromosome'],
                start=context_start,
                end=context_end,
                organism=organism,
                run_name=f"{base_name}_score",
                output_dir=output_dir,
                gene_info=gene_info
            )

        # Create baseline report
        baseline = {
            'gene': gene,
            'gene_info': gene_info,
            'cell_type': cell_type,
            'organism': organism,
            'context': {
                'chromosome': gene_info['chromosome'],
                'start': context_start,
                'end': context_end,
                'window_size': window
            },
            'analyses': {
                'prediction': f"{base_name}_predict.npz",
                'scoring': f"{base_name}_score.json" if window >= 524288 else None
            }
        }

        # Save baseline
        output_path = Path(output_dir or 'results')
        output_path.mkdir(parents=True, exist_ok=True)
        baseline_file = output_path / f"{base_name}_report.json"

        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)

        print(f"\n✅ Baseline report saved to {base_name}_report.json")
        print(f"\n{'='*60}")
        print("Analysis complete! Visualize with:")
        print(f"  python visualizer.py {output_path}/{base_name}_predict.npz --tracks all --overlays tss gene promoter enhancer --peaks")
        print(f"  python npz_viewer.py {output_path}/{base_name}_predict.npz")
        print('='*60)

        return baseline


# Command-line interface
def show_available_options(option_type, ag=None):
    """Show available organisms, cell types, or tissues"""

    if ag is None:
        ag = AlphaGenome()

    if option_type == 'organisms':
        print("\nAvailable Organisms:")
        print("="*60)
        organisms = ag.get_available_organisms()
        for code, name in organisms.items():
            print(f"  {code:10} : {name}")

    elif option_type == 'cell-types' or option_type == 'ontologies':
        print("\nAvailable Cell Types:")
        print("="*60)
        cell_types = ag.get_available_cell_types()
        for ont_id, name in sorted(cell_types.items()):
            print(f"  {ont_id:12} : {name}")

    elif option_type == 'tissues':
        print("\nAvailable Tissues:")
        print("="*60)
        tissues = ag.get_available_tissues()
        for ont_id, name in sorted(tissues.items()):
            print(f"  {ont_id:14} : {name}")

    else:
        print(f"Unknown option type: {option_type}")
        print("Valid options: organisms, cell-types, tissues")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='AlphaGenome API Interface - Predict, score, and visualize regulatory activity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Predict using gene symbol (NEW - automatically fetches coordinates!)
  %(prog)s --method predict_interval --gene GFAP
  %(prog)s --method predict_interval --gene GFAP --window 131072  # 131kb window
  %(prog)s --method predict_interval --gene Gfap --organism mouse  # Mouse gene

  # Predict regulatory tracks for genomic interval
  %(prog)s --method predict_interval --chromosome chr17 --coords 42322400-42338784

  # Predict from DNA sequence file
  %(prog)s --method predict_sequence --sequence sequence.fasta --ontology CL:0000127

  # Predict variant effects
  %(prog)s --method predict_variant --variant chr17:42322400:A:G

  # Score variant impact
  %(prog)s --method score_variant --variant chr17:42322400:A:G --organism human

  # Score interval activity
  %(prog)s --method score_interval --chromosome chr1 --coords 1000000-1524288

  # In-Silico Mutagenesis (ISM)
  %(prog)s --method score_ism --chromosome chr17 --ism-coords 42322400-42322410

  # Gene regulatory baseline (replaces get_gene_ground_truth.py)
  %(prog)s --method gene_baseline --gene GFAP --cell-type CL:0000127
  %(prog)s --method gene_baseline --gene TP53 --window 1048576  # 1Mb window

Supported methods:
  predict_interval  : Predict regulatory tracks for genomic interval
  predict_sequence  : Predict tracks from DNA sequence
  predict_variant   : Predict variant effect on regulatory tracks
  score_variant     : Score variant impact with multiple scorers
  score_interval    : Score regulatory activity across interval
  score_ism         : In-Silico Mutagenesis - test all mutations in region
  gene_baseline     : Baseline regulatory analysis for a gene

Valid interval sizes: 16384, 131072, 524288, 1048576 bp

Cell type ontologies (examples):
  CL:0000127 : Astrocyte (default)
  CL:0000746 : Cardiac muscle cell
  CL:0002322 : Embryonic stem cell
  CL:0000057 : Fibroblast
  CL:0000084 : T cell

Output files:
  - .npz  : Compressed numpy arrays with predictions/scores
  - .json : Metadata and configuration

Note: Use visualizer.py to create publication-quality visualizations from NPZ files:
  python visualizer.py results/output.npz --tracks dnase rna-seq --peaks
        '''
    )

    # Special help options
    parser.add_argument('--list-organisms', action='store_true',
                       help='Show available organisms')
    parser.add_argument('--list-cell-types', '--list-ontologies', action='store_true',
                       dest='list_cell_types', help='Show available cell type ontologies')
    parser.add_argument('--list-tissues', action='store_true',
                       help='Show available tissue ontologies')

    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')

    # Method selection
    parser.add_argument('--method', choices=[
        'predict_interval', 'predict_sequence', 'predict_variant',
        'score_variant', 'score_interval', 'score_ism', 'gene_baseline'
    ], help='API method to call (required unless using --list-* options)')

    # Common arguments
    parser.add_argument('--gene', metavar='SYMBOL',
                       help='Gene symbol (e.g., GFAP, TP53) - automatically fetches coordinates')
    parser.add_argument('--chromosome', metavar='CHR',
                       help='Chromosome name (e.g., chr17, chr1, chrX)')
    parser.add_argument('--organism', default='human', choices=['human', 'mouse'],
                       help='Organism (default: human)')
    parser.add_argument('--window', type=int, choices=[16384, 131072, 524288, 1048576],
                       default=524288,
                       help='Window size for gene context (default: 524288bp/524kb). Note: score_interval requires >=524kb')
    parser.add_argument('--ontology', '--cell-type', default=DEFAULT_ONTOLOGY,
                       metavar='ONTOLOGY_ID', dest='ontology',
                       help=f'Cell type ontology ID (default: {DEFAULT_ONTOLOGY} - astrocyte)')
    parser.add_argument('--run-name', metavar='NAME',
                       help='Custom name for output files (default: auto-generated)')
    parser.add_argument('--output-dir', metavar='DIR', default='results',
                       help='Output directory (default: results/)')

    # Method-specific arguments
    parser.add_argument('--coords', metavar='START-END',
                       help='Genomic coordinates as start-end (e.g., 1000000-1131072)')
    parser.add_argument('--sequence', metavar='SEQ_OR_FILE',
                       help='DNA sequence string or path to FASTA file')
    parser.add_argument('--variant', metavar='CHR:POS:REF:ALT',
                       help='Variant in format chr:position:ref:alt (e.g., chr17:42322400:A:G)')
    parser.add_argument('--context-start', type=int, metavar='POS',
                       help='Context window start position (for variant/ISM methods)')
    parser.add_argument('--context-end', type=int, metavar='POS',
                       help='Context window end position (for variant/ISM methods)')
    parser.add_argument('--ism-coords', metavar='START-END',
                       help='ISM region coordinates as start-end (e.g., 42322400-42322410)')

    args = parser.parse_args()

    # Handle special list options
    if hasattr(args, 'list_organisms') and args.list_organisms:
        show_available_options('organisms')
        sys.exit(0)

    if hasattr(args, 'list_cell_types') and args.list_cell_types:
        show_available_options('cell-types')
        sys.exit(0)

    if hasattr(args, 'list_tissues') and args.list_tissues:
        show_available_options('tissues')
        sys.exit(0)

    # Handle gene symbol to coordinates conversion
    if args.gene:
        import requests

        # Determine species for Ensembl API
        species = 'homo_sapiens' if args.organism == 'human' else 'mus_musculus'

        print(f"Fetching coordinates for {args.gene} ({args.organism})...")
        server = "https://rest.ensembl.org"
        ext = f"/lookup/symbol/{species}/{args.gene}?expand=1"

        try:
            r = requests.get(server + ext, headers={"Content-Type": "application/json"})
            if r.ok:
                data = r.json()
                args.chromosome = f"chr{data.get('seq_region_name')}"
                gene_start = data.get('start')
                gene_end = data.get('end')

                # Save gene coordinates for later use
                args._gene_start = gene_start
                args._gene_end = gene_end

                # Calculate appropriate window based on method
                if args.method in ['predict_interval', 'score_interval']:
                    # Check window size requirement for score_interval
                    if args.method == 'score_interval' and args.window < 524288:
                        print(f"  ⚠️  Warning: score_interval requires window >= 524kb")
                        print(f"     Using minimum 524kb window instead of {args.window // 1024}kb")
                        args.window = 524288

                    # Create window centered on gene using user-specified size
                    gene_center = (gene_start + gene_end) // 2
                    half_window = args.window // 2
                    window_start = max(0, gene_center - half_window)
                    window_end = gene_center + half_window
                    args.coords = f"{window_start}-{window_end}"

                    # Format window size for display
                    if args.window >= 1048576:
                        window_display = f"{args.window // 1048576}Mb"
                    else:
                        window_display = f"{args.window // 1024}kb"

                    print(f"  Gene: {args.chromosome}:{gene_start:,}-{gene_end:,}")
                    print(f"  Context: {args.chromosome}:{window_start:,}-{window_end:,} ({window_display})")
                elif args.method == 'predict_variant':
                    # For variants, use gene start as position if not specified
                    if not args.variant:
                        print(f"  Gene: {args.chromosome}:{gene_start:,}-{gene_end:,}")
                        print(f"  Note: For variants, also specify --variant")

                if not args.run_name:
                    args.run_name = args.gene
            else:
                print(f"❌ Could not find gene {args.gene}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error fetching gene: {e}")
            sys.exit(1)

    # Initialize AlphaGenome
    ag = AlphaGenome(verbose=args.verbose)

    # Route to appropriate method
    if args.method == 'predict_interval':
        if not args.chromosome or not args.coords:
            parser.error("predict_interval requires --chromosome and --coords")
        start, end = map(int, args.coords.split('-'))
        ag.predict_interval(
            chromosome=args.chromosome, start=start, end=end,
            ontology=args.ontology, organism=args.organism,
            run_name=args.run_name, output_dir=args.output_dir
        )

    elif args.method == 'predict_sequence':
        if not args.sequence:
            parser.error("predict_sequence requires --sequence")
        ag.predict_sequence(
            sequence=args.sequence, ontology=args.ontology,
            organism=args.organism, run_name=args.run_name,
            output_dir=args.output_dir
        )

    elif args.method == 'predict_variant':
        if not args.variant:
            parser.error("predict_variant requires --variant")
        parts = args.variant.split(':')
        if len(parts) != 4:
            parser.error("Invalid variant format. Use chr:pos:ref:alt")
        chromosome, position, ref, alt = parts
        ag.predict_variant(
            chromosome=chromosome, position=int(position),
            ref=ref, alt=alt,
            context_start=args.context_start, context_end=args.context_end,
            ontology=args.ontology, organism=args.organism,
            run_name=args.run_name, output_dir=args.output_dir
        )

    elif args.method == 'score_variant':
        if not args.variant:
            parser.error("score_variant requires --variant")
        parts = args.variant.split(':')
        if len(parts) != 4:
            parser.error("Invalid variant format. Use chr:pos:ref:alt")
        chromosome, position, ref, alt = parts
        ag.score_variant(
            chromosome=chromosome, position=int(position),
            ref=ref, alt=alt,
            context_start=args.context_start, context_end=args.context_end,
            organism=args.organism, run_name=args.run_name,
            output_dir=args.output_dir
        )

    elif args.method == 'score_interval':
        if not args.chromosome or not args.coords:
            parser.error("score_interval requires --chromosome and --coords")
        start, end = map(int, args.coords.split('-'))

        # Pass gene info if we fetched it
        gene_info = None
        if args.gene and hasattr(args, '_gene_start'):
            gene_info = {
                'name': args.gene,
                'start': args._gene_start,
                'end': args._gene_end
            }

        ag.score_interval(
            chromosome=args.chromosome, start=start, end=end,
            organism=args.organism, run_name=args.run_name,
            output_dir=args.output_dir, gene_info=gene_info
        )

    elif args.method == 'score_ism':
        if not args.chromosome or not args.ism_coords:
            parser.error("score_ism requires --chromosome and --ism_coords")
        ism_start, ism_end = map(int, args.ism_coords.split('-'))

        # Warn if large region
        ism_size = ism_end - ism_start
        if ism_size > 200:
            print(f"⚠️  Warning: ISM region is {ism_size} bp ({ism_size * 3} mutations)")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)

        ag.score_ism_variants(
            chromosome=args.chromosome, ism_start=ism_start, ism_end=ism_end,
            context_start=args.context_start, context_end=args.context_end,
            organism=args.organism, run_name=args.run_name,
            output_dir=args.output_dir
        )

    elif args.method == 'gene_baseline':
        if not args.gene:
            parser.error("gene_baseline requires --gene")
        ag.gene_baseline(
            gene=args.gene,
            cell_type=args.ontology,
            organism=args.organism,
            window=args.window or 524288,
            run_name=args.run_name,
            output_dir=args.output_dir
        )

    else:
        parser.error("Please specify a --method")


if __name__ == "__main__":
    main()