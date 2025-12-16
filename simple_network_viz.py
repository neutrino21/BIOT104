#!/usr/bin/env python3
"""
Simple Network Visualization - No threading issues

-----------------------------------------------------------------------
Copyright (c) 2025 Sean Kiewiet. All rights reserved.
-----------------------------------------------------------------------
"""

import http.server
import socketserver
import json
import sqlite3
import os

PORT = 8000

class NetworkHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('network_viz.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path.startswith('/api/search/'):
            parts = self.path.split('/')
            query = parts[-1].upper()
            depth = 2  # default depth

            # Check if depth is specified
            if '?' in query:
                query, params = query.split('?')
                if 'depth=' in params:
                    depth = int(params.split('=')[1])

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # Connect to database
            conn = sqlite3.connect('tf_gene_network.db')
            cursor = conn.cursor()

            # Function to get multi-level connections
            def get_multilevel(start_node, direction, max_depth):
                levels = {1: set()}

                # Get first level
                if direction == 'upstream':
                    first = cursor.execute(
                        "SELECT source FROM edges WHERE target = ?",
                        (start_node,)
                    ).fetchall()
                    levels[1] = set([r[0] for r in first])
                else:
                    first = cursor.execute(
                        "SELECT target FROM edges WHERE source = ?",
                        (start_node,)
                    ).fetchall()
                    levels[1] = set([t[0] for t in first])

                # Get additional levels
                for d in range(2, max_depth + 1):
                    levels[d] = set()
                    for node in levels[d-1]:
                        if direction == 'upstream':
                            next_level = cursor.execute(
                                "SELECT source FROM edges WHERE target = ?",
                                (node,)
                            ).fetchall()
                            levels[d].update([r[0] for r in next_level])
                        else:
                            next_level = cursor.execute(
                                "SELECT target FROM edges WHERE source = ?",
                                (node,)
                            ).fetchall()
                            levels[d].update([t[0] for t in next_level])

                return levels

            # Get multi-level upstream and downstream
            upstream_levels = get_multilevel(query, 'upstream', depth)
            downstream_levels = get_multilevel(query, 'downstream', depth)

            # Get node info
            node_info = cursor.execute(
                "SELECT is_gene, is_tf, is_memory_gene FROM nodes WHERE name = ?",
                (query,)
            ).fetchone()

            conn.close()

            # Prepare result with levels - no limits
            result = {
                'node': query,
                'depth': depth,
                'upstream_levels': {str(k): list(v) for k, v in upstream_levels.items()},
                'downstream_levels': {str(k): list(v) for k, v in downstream_levels.items()},
                'upstream': list(upstream_levels.get(1, [])),
                'downstream': list(downstream_levels.get(1, [])),
                'info': {
                    'is_gene': bool(node_info[0]) if node_info else False,
                    'is_tf': bool(node_info[1]) if node_info else False,
                    'is_memory': bool(node_info[2]) if node_info else False
                }
            }

            self.wfile.write(json.dumps(result).encode())
        else:
            super().do_GET()

# Create HTML file
HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>TF-Gene Network Search</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .search-bar {
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }
        input {
            flex: 1;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        button {
            padding: 12px 30px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #45a049;
        }
        #results {
            margin-top: 20px;
        }
        .info-box {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .section {
            margin: 20px 0;
        }
        .section h3 {
            color: #555;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .node-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .node-item {
            padding: 8px;
            background: #e8f5e9;
            border: 1px solid #4CAF50;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .node-item:hover {
            background: #c8e6c9;
            transform: scale(1.05);
        }
        .node-item.tf {
            background: #fff3e0;
            border-color: #ff9800;
        }
        .node-item.tf:hover {
            background: #ffe0b2;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            background: #666;
            color: white;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 5px;
        }
        .badge.gene { background: #4CAF50; }
        .badge.tf { background: #ff9800; }
        .badge.memory { background: #f44336; }
        #network-viz {
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stats {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
        }
        .stat-card {
            padding: 15px 25px;
            background: #f5f5f5;
            border-radius: 5px;
            text-align: center;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>TF-Gene Regulatory Network Explorer</h1>

        <div class="search-bar">
            <input type="text" id="search-input" placeholder="Enter gene or TF name (e.g., VEGFA, STAT3, IL1B)..." />
            <select id="direction-select">
                <option value="both">Both Directions</option>
                <option value="upstream">Upstream Only</option>
                <option value="downstream">Downstream Only</option>
            </select>
            <select id="depth-select">
                <option value="1">1 Level</option>
                <option value="2" selected>2 Levels</option>
                <option value="3">3 Levels</option>
            </select>
            <button onclick="searchNetwork()">Search</button>
        </div>

        <div id="results"></div>
    </div>

    <script>
        let currentSvg = null;
        let currentZoom = null;
        let currentG = null;

        function zoomIn() {
            if (currentSvg && currentZoom) {
                currentSvg.transition().duration(300).call(currentZoom.scaleBy, 1.3);
            }
        }

        function zoomOut() {
            if (currentSvg && currentZoom) {
                currentSvg.transition().duration(300).call(currentZoom.scaleBy, 0.7);
            }
        }

        function resetZoom() {
            if (currentSvg && currentZoom) {
                currentSvg.transition().duration(300).call(currentZoom.transform, d3.zoomIdentity);
            }
        }

        function fitToScreen() {
            if (!currentG || !currentSvg) return;

            const bounds = currentG.node().getBBox();
            const fullWidth = window.innerWidth - 100;
            const fullHeight = 800;
            const width = bounds.width;
            const height = bounds.height;
            const midX = bounds.x + width / 2;
            const midY = bounds.y + height / 2;

            const scale = 0.9 / Math.max(width / fullWidth, height / fullHeight);
            const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];

            currentSvg.transition().duration(500).call(
                currentZoom.transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
        }

        async function searchNetwork() {
            const query = document.getElementById('search-input').value.trim().toUpperCase();
            if (!query) return;

            const depth = document.getElementById('depth-select').value;
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p>Loading...</p>';

            try {
                const response = await fetch(`/api/search/${query}?depth=${depth}`);
                const data = await response.json();

                let html = `
                    <div class="info-box">
                        <h2>${data.node}</h2>
                        <div class="badges">
                            ${data.info.is_gene ? '<span class="badge gene">Gene</span>' : ''}
                            ${data.info.is_tf ? '<span class="badge tf">TF</span>' : ''}
                            ${data.info.is_memory ? '<span class="badge memory">Memory Gene</span>' : ''}
                        </div>
                    </div>

                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">${data.upstream.length}</div>
                            <div class="stat-label">Upstream Regulators</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.downstream.length}</div>
                            <div class="stat-label">Downstream Targets</div>
                        </div>
                    </div>
                `;

                const direction = document.getElementById('direction-select').value;

                if (data.upstream.length > 0 && (direction === 'both' || direction === 'upstream')) {
                    html += `
                        <div class="section">
                            <h3>Upstream Regulators (TFs that regulate ${data.node}) - Total: ${data.upstream.length}</h3>
                            <div class="node-list">
                                ${data.upstream.map(tf =>
                                    `<div class="node-item tf" onclick="document.getElementById('search-input').value='${tf}'; searchNetwork();">${tf}</div>`
                                ).join('')}
                            </div>
                        </div>
                    `;
                }

                if (data.downstream.length > 0 && (direction === 'both' || direction === 'downstream')) {
                    html += `
                        <div class="section">
                            <h3>Downstream Targets (Genes/TFs regulated by ${data.node}) - Total: ${data.downstream.length}</h3>
                            <div class="node-list">
                                ${data.downstream.map(gene =>
                                    `<div class="node-item" onclick="document.getElementById('search-input').value='${gene}'; searchNetwork();">${gene}</div>`
                                ).join('')}
                            </div>
                        </div>
                    `;
                }

                if (data.upstream.length === 0 && data.downstream.length === 0) {
                    html += '<p>No regulatory connections found for this node.</p>';
                }

                // Add network visualization with controls
                html += `
                    <div style="margin: 10px 0;">
                        <button onclick="zoomIn()" style="margin-right: 5px;">Zoom In</button>
                        <button onclick="zoomOut()" style="margin-right: 5px;">Zoom Out</button>
                        <button onclick="resetZoom()" style="margin-right: 5px;">Reset Zoom</button>
                        <button onclick="fitToScreen()">Fit to Screen</button>
                    </div>
                    <div id="network-viz"></div>
                `;

                resultsDiv.innerHTML = html;

                // Create simple network visualization
                if (data.upstream.length > 0 || data.downstream.length > 0) {
                    createNetworkViz(data);
                }

            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }

        function createNetworkViz(data) {
            const width = window.innerWidth - 100;
            const height = 800;
            const direction = document.getElementById('direction-select').value;

            // Create nodes and links
            const nodeMap = {};
            const nodes = [];
            const links = [];

            // Add root node
            nodeMap[data.node] = {id: data.node, group: 'root', level: 0};
            nodes.push(nodeMap[data.node]);

            // Add multi-level upstream nodes
            if (data.upstream_levels && (direction === 'both' || direction === 'upstream')) {
                const selectedDepth = parseInt(document.getElementById('depth-select').value);
                for (let level = 1; level <= selectedDepth; level++) {
                    const levelNodes = data.upstream_levels[level.toString()] || [];
                    levelNodes.forEach((node, i) => {
                        if (!nodeMap[node]) {
                            nodeMap[node] = {
                                id: node,
                                group: 'upstream',
                                level: -level
                            };
                            nodes.push(nodeMap[node]);
                        }
                    });

                    // Add links - only for nodes we're displaying
                    if (level === 1) {
                        levelNodes.forEach(node => {
                            if (nodeMap[node]) {
                                links.push({source: node, target: data.node});
                            }
                        });
                    } else if (level <= selectedDepth) {
                        // Only add links for levels we're actually showing
                        const prevLevel = data.upstream_levels[(level-1).toString()] || [];
                        levelNodes.forEach(node => {
                            if (nodeMap[node] && prevLevel.length > 0) {
                                // Connect to a node from previous level that exists
                                const validPrevNodes = prevLevel.filter(n => nodeMap[n]);
                                if (validPrevNodes.length > 0) {
                                    const prevNode = validPrevNodes[Math.floor(Math.random() * Math.min(3, validPrevNodes.length))];
                                    links.push({source: node, target: prevNode});
                                }
                            }
                        });
                    }
                }
            }

            // Add multi-level downstream nodes
            if (data.downstream_levels && (direction === 'both' || direction === 'downstream')) {
                const selectedDepth = parseInt(document.getElementById('depth-select').value);
                for (let level = 1; level <= selectedDepth; level++) {
                    const levelNodes = data.downstream_levels[level.toString()] || [];
                    levelNodes.forEach((node, i) => {
                        if (!nodeMap[node]) {
                            nodeMap[node] = {
                                id: node,
                                group: 'downstream',
                                level: level
                            };
                            nodes.push(nodeMap[node]);
                        }
                    });

                    // Add links - only for nodes we're displaying
                    if (level === 1) {
                        levelNodes.forEach(node => {
                            if (nodeMap[node]) {
                                links.push({source: data.node, target: node});
                            }
                        });
                    } else if (level <= selectedDepth) {
                        // Only add links for levels we're actually showing
                        const prevLevel = data.downstream_levels[(level-1).toString()] || [];
                        levelNodes.forEach(node => {
                            if (nodeMap[node] && prevLevel.length > 0) {
                                // Connect from a node in previous level that exists
                                const validPrevNodes = prevLevel.filter(n => nodeMap[n]);
                                if (validPrevNodes.length > 0) {
                                    const prevNode = validPrevNodes[Math.floor(Math.random() * Math.min(3, validPrevNodes.length))];
                                    links.push({source: prevNode, target: node});
                                }
                            }
                        });
                    }
                }
            }

            // Filter out nodes with no connections
            const connectedNodes = new Set();
            links.forEach(link => {
                connectedNodes.add(link.source);
                connectedNodes.add(link.target);
            });

            const filteredNodes = nodes.filter(node => connectedNodes.has(node.id) || node.group === 'root');

            // Clear previous visualization
            d3.select("#network-viz").selectAll("*").remove();

            // Create visualization
            const svg = d3.select("#network-viz")
                .append("svg")
                .attr("width", width)
                .attr("height", height);

            // Store references globally
            currentSvg = svg;

            // Add zoom
            const g = svg.append("g");
            currentG = g;

            const zoom = d3.zoom()
                .scaleExtent([0.05, 10])
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                });

            currentZoom = zoom;
            svg.call(zoom);

            // Position nodes by level (use filtered nodes)
            filteredNodes.forEach(node => {
                if (node.level === 0) {
                    node.fx = width / 2;
                    node.fy = height / 2;
                } else {
                    // Position based on level with more spread
                    const levelX = width / 2 + (node.level * 200);
                    node.x = levelX;
                    node.y = height / 2 + (Math.random() - 0.5) * 400;
                }
            });

            // Create force simulation (use filtered nodes)
            const simulation = d3.forceSimulation(filteredNodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(120))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("x", d3.forceX(d => width/2 + d.level * 200).strength(0.4))
                .force("y", d3.forceY(height/2).strength(0.05))
                .force("collision", d3.forceCollide().radius(30));

            // Add links
            const link = g.append("g")
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .style("stroke", "#999")
                .style("stroke-opacity", 0.6)
                .style("stroke-width", 2);

            // Add nodes (use filtered nodes)
            const node = g.append("g")
                .selectAll("circle")
                .data(filteredNodes)
                .enter().append("circle")
                .attr("r", d => d.group === 'root' ? 12 : 6 + Math.abs(d.level))
                .style("fill", d => {
                    if (d.group === 'root') return "#ff4444";
                    if (d.group === 'upstream') {
                        // Gradient based on level
                        const opacity = 1 - (Math.abs(d.level) - 1) * 0.3;
                        return `rgba(68, 255, 68, ${opacity})`;
                    }
                    // Downstream
                    const opacity = 1 - (d.level - 1) * 0.3;
                    return `rgba(68, 68, 255, ${opacity})`;
                })
                .style("stroke", "#fff")
                .style("stroke-width", 2)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Add labels (use filtered nodes)
            const label = g.append("g")
                .selectAll("text")
                .data(filteredNodes)
                .enter().append("text")
                .text(d => d.id)
                .style("font-size", "10px")
                .style("text-anchor", "middle")
                .attr("dy", -12);

            // Add click to search
            node.on("click", (event, d) => {
                if (d.id !== data.node) {
                    document.getElementById('search-input').value = d.id;
                    searchNetwork();
                }
            });

            // Update positions
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            });

            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }

        // Handle Enter key
        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchNetwork();
            }
        });

        // Example search
        document.getElementById('search-input').value = 'VEGFA';
        searchNetwork();
    </script>
</body>
</html>'''

# Write HTML file
with open('network_viz.html', 'w') as f:
    f.write(HTML)

print("Starting server at http://localhost:8000")
print("Press Ctrl+C to stop")

# Start server
with socketserver.TCPServer(("", PORT), NetworkHandler) as httpd:
    httpd.serve_forever()