#!/usr/bin/env python3
"""
TF Network API - Unified interface for querying TF regulatory networks

Provides fast, efficient access to multi-depth TF regulatory networks

-----------------------------------------------------------------------
Copyright (c) 2025 Sean Kiewiet. All rights reserved.
-----------------------------------------------------------------------
built from merged TRRUST + promoter GRN data.

Author: CellOracle Team
Version: 1.0.0
"""

import sqlite3
import networkx as nx
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict


class TFNetworkAPI:
    """
    Main API for querying TF regulatory networks.
    
    Supports multiple backends:
    - SQLite for complex queries
    - NetworkX for path finding
    - NPZ for matrix operations
    
    Examples:
        >>> api = TFNetworkAPI()
        >>> api.get_regulators('IL1B')
        ['AHR', 'CEBPB', 'RELA', ...]
        
        >>> api.find_path('IL1B', 'STAT3')
        ['IL1B', 'NFKB1', 'STAT3']
    """
    
    def __init__(self,
                 db_path: str = 'tf_gene_network.db',
                 graph_path: str = 'tf_gene_network.gml',
                 npz_path: str = 'tf_gene_network.npz',
                 use_cache: bool = True):
        """
        Initialize TF Network API.
        
        Args:
            db_path: Path to SQLite database
            graph_path: Path to NetworkX graph file
            npz_path: Path to NPZ array file
            use_cache: Whether to cache query results
        """
        self.db_path = db_path
        self.graph_path = graph_path
        self.npz_path = npz_path
        self.use_cache = use_cache
        
        # Initialize connections
        self.conn = None
        self.cursor = None
        self.graph = None
        self.npz_data = None
        
        # Cache for frequent queries
        self._cache = {} if use_cache else None
        
        # Load database
        self._connect_db()
        
    def _connect_db(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def _load_graph(self):
        """Lazy load NetworkX graph."""
        if self.graph is None:
            self.graph = nx.read_gml(self.graph_path)
        return self.graph
    
    def _load_npz(self):
        """Lazy load NPZ data."""
        if self.npz_data is None:
            self.npz_data = np.load(self.npz_path, allow_pickle=True)
        return self.npz_data
    
    # ============================================================
    # Core Query Methods
    # ============================================================
    
    def get_regulators(self, gene: str) -> List[str]:
        """
        Get TFs that regulate a gene.

        Args:
            gene: Gene name

        Returns:
            List of regulating TFs
        """
        cache_key = f"reg_{gene}"
        if self._cache and cache_key in self._cache:
            return self._cache[cache_key]

        result = self.cursor.execute(
            "SELECT DISTINCT source FROM edges WHERE target = ?",
            (gene,)
        ).fetchall()

        regulators = [r[0] for r in result]

        if self._cache:
            self._cache[cache_key] = regulators

        return regulators
    
    def get_targets(self, tf: str) -> List[str]:
        """
        Get genes regulated by a TF.

        Args:
            tf: TF name

        Returns:
            List of target genes
        """
        cache_key = f"tar_{tf}"
        if self._cache and cache_key in self._cache:
            return self._cache[cache_key]

        result = self.cursor.execute(
            "SELECT DISTINCT target FROM edges WHERE source = ?",
            (tf,)
        ).fetchall()

        targets = [r[0] for r in result]

        if self._cache:
            self._cache[cache_key] = targets

        return targets
    
    def find_path(self, source: str, target: str, 
                  shortest: bool = True,
                  max_length: int = 3) -> Union[List[str], List[List[str]]]:
        """
        Find path(s) between two nodes.
        
        Args:
            source: Starting node
            target: Ending node
            shortest: Return only shortest path
            max_length: Maximum path length
            
        Returns:
            Path or list of paths
        """
        G = self._load_graph()
        
        try:
            if shortest:
                return nx.shortest_path(G, source, target)
            else:
                paths = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
                return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [] if not shortest else None
    
    def get_cascade(self, gene: str, max_depth: int = 3) -> Dict[str, List[str]]:
        """
        Get full regulatory cascade from a gene.
        
        Args:
            gene: Starting gene
            max_depth: Maximum cascade depth
            
        Returns:
            Dictionary with nodes at each depth
        """
        G = self._load_graph()
        cascade = defaultdict(set)
        
        if gene not in G:
            return {}
        
        # BFS to find cascade
        queue = [(gene, 0)]
        visited = {gene}
        
        while queue:
            node, depth = queue.pop(0)
            
            if depth > max_depth:
                continue
                
            cascade[f"depth_{depth}"].add(node)
            
            for successor in G.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, depth + 1))
        
        # Convert to sorted lists
        result = {}
        for depth_key, nodes in cascade.items():
            result[depth_key] = sorted(nodes)
            
        return result
    
    def find_common_regulators(self, genes: List[str]) -> List[str]:
        """
        Find TFs that regulate multiple genes.
        
        Args:
            genes: List of gene names
            
        Returns:
            List of common regulating TFs
        """
        if not genes:
            return []
        
        # Get regulators for each gene
        regulator_sets = []
        for gene in genes:
            regulators = set(self.get_regulators(gene))
            regulator_sets.append(regulators)
        
        # Find intersection
        common = regulator_sets[0]
        for reg_set in regulator_sets[1:]:
            common = common.intersection(reg_set)
        
        return sorted(common)
    
    def find_hub_nodes(self, top_n: int = 10, node_type: str = 'both') -> Dict:
        """
        Find most connected nodes (hubs).
        
        Args:
            top_n: Number of top nodes to return
            node_type: 'regulators', 'targets', or 'both'
            
        Returns:
            Dictionary with hub nodes and their degrees
        """
        result = {}
        
        if node_type in ['regulators', 'both']:
            # Nodes with most outgoing edges
            query = """
                SELECT source, COUNT(*) as out_degree
                FROM edges
                GROUP BY source
                ORDER BY out_degree DESC
                LIMIT ?
            """
            regulators = self.cursor.execute(query, (top_n,)).fetchall()
            result['top_regulators'] = [(n, d) for n, d in regulators]
        
        if node_type in ['targets', 'both']:
            # Nodes with most incoming edges
            query = """
                SELECT target, COUNT(*) as in_degree
                FROM edges
                GROUP BY target
                ORDER BY in_degree DESC
                LIMIT ?
            """
            targets = self.cursor.execute(query, (top_n,)).fetchall()
            result['most_regulated'] = [(n, d) for n, d in targets]
        
        return result
    
    def get_node_info(self, node: str) -> Optional[Dict]:
        """
        Get detailed information about a node.
        
        Args:
            node: Node name
            
        Returns:
            Dictionary with node information
        """
        # Check if node exists
        result = self.cursor.execute(
            "SELECT * FROM nodes WHERE name = ?",
            (node,)
        ).fetchone()
        
        if not result:
            return None
        
        info = {
            'name': node,
            'type': result[2],
            'is_memory_gene': bool(result[3]),
            'is_tf': bool(result[4])
        }
        
        # Get connectivity
        info['regulators'] = self.get_regulators(node)
        info['targets'] = self.get_targets(node)
        info['in_degree'] = len(info['regulators'])
        info['out_degree'] = len(info['targets'])
        
        return info
    
    def find_motifs(self, motif_type: str = 'feedforward', max_size: int = 3) -> List:
        """
        Find regulatory motifs in the network.
        
        Args:
            motif_type: 'feedforward', 'feedback', or 'all'
            max_size: Maximum motif size
            
        Returns:
            List of motifs
        """
        G = self._load_graph()
        motifs = []
        
        if motif_type in ['feedforward', 'all']:
            # Find feed-forward loops (A->B, A->C, B->C)
            for a in G.nodes():
                a_targets = set(G.successors(a))
                
                for b in a_targets:
                    b_targets = set(G.successors(b))
                    
                    # Common targets of A and B
                    common = a_targets.intersection(b_targets)
                    
                    for c in common:
                        motifs.append(('feedforward', [a, b, c]))
        
        if motif_type in ['feedback', 'all']:
            # Find feedback loops (cycles)
            for cycle in nx.simple_cycles(G):
                if len(cycle) <= max_size:
                    motifs.append(('feedback', cycle))
        
        return motifs
    
    def get_memory_genes(self) -> List[str]:
        """
        Get all memory genes in the network.
        
        Returns:
            List of memory gene names
        """
        result = self.cursor.execute(
            "SELECT name FROM nodes WHERE is_memory_gene = 1 ORDER BY name"
        ).fetchall()
        
        return [r[0] for r in result]
    
    def get_all_tfs(self) -> List[str]:
        """
        Get all TFs in the network.
        
        Returns:
            List of TF names
        """
        result = self.cursor.execute(
            "SELECT name FROM nodes WHERE is_tf = 1 ORDER BY name"
        ).fetchall()
        
        return [r[0] for r in result]
    
    def get_network_stats(self) -> Dict:
        """
        Get comprehensive network statistics.
        
        Returns:
            Dictionary with network metrics
        """
        stats = {}
        
        # Node counts
        stats['total_nodes'] = self.cursor.execute(
            "SELECT COUNT(*) FROM nodes"
        ).fetchone()[0]
        
        stats['memory_genes'] = self.cursor.execute(
            "SELECT COUNT(*) FROM nodes WHERE is_memory_gene = 1"
        ).fetchone()[0]
        
        stats['tfs'] = self.cursor.execute(
            "SELECT COUNT(*) FROM nodes WHERE is_tf = 1"
        ).fetchone()[0]
        
        # Edge counts by depth
        for depth in [1, 2, 3]:
            count = self.cursor.execute(
                "SELECT COUNT(*) FROM edges WHERE depth = ?",
                (depth,)
            ).fetchone()[0]
            stats[f'edges_depth_{depth}'] = count
        
        stats['total_edges'] = self.cursor.execute(
            "SELECT COUNT(*) FROM edges"
        ).fetchone()[0]
        
        # Network density
        if stats['total_nodes'] > 1:
            max_edges = stats['total_nodes'] * (stats['total_nodes'] - 1)
            stats['density'] = stats['total_edges'] / max_edges if max_edges > 0 else 0
        
        return stats
    
    def export_subgraph(self, nodes: List[str], 
                       output_file: str = 'subgraph.gml') -> nx.DiGraph:
        """
        Export subgraph containing specified nodes.
        
        Args:
            nodes: List of node names
            output_file: Output file path
            
        Returns:
            NetworkX subgraph
        """
        G = self._load_graph()
        subgraph = G.subgraph(nodes)
        nx.write_gml(subgraph, output_file)
        return subgraph
    
    def query(self, sql: str, params: Tuple = ()) -> List:
        """
        Execute custom SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        return self.cursor.execute(sql, params).fetchall()
    
    def close(self):
        """Close all connections."""
        if self.conn:
            self.conn.close()
        self.graph = None
        self.npz_data = None
        if self._cache:
            self._cache.clear()