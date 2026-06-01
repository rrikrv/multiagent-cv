# multiagent-cv
Computer Vision module for multi-agent planning platform


# Computer Vision Module for Multi-Agent Planning

A simulation system for navigating multiple autonomous agents 
across polygonal maps with obstacles. Developed as part of the 
HSE University team project "Platform for Multi-Agent Planning".

## Overview

The system implements:
- **A\* pathfinding** with Manhattan heuristic for optimal navigation
- **Three-class object classification**: background, static obstacles, dynamic agents
- **Conflict resolution**: cell reservation, swap handling, oscillation detection
- **Configurable Field of View (FoV)** simulating real sensor limitations
- **Universal polygon support**: works with any map in standard format

## Project Structure

    multiagent_cv/
    ├── data/
    │   ├── warehouse/      # MAPD scenario
    │   ├── maze/           # MAPF scenario
    │   ├── bottleneck/     # MAPF scenario
    │   └── thin_walls/     # MAPF scenario
    ├── src/
    │   ├── main.py         # Entry point
    │   ├── simulator.py    # A* navigation and conflict resolution
    │   ├── map_parser.py   # PNG map binarization
    │   └── cv_module.py    # CV classification and visualization
    └── requirements.txt
## Installation

```bash
pip3 install -r requirements.txt
```

## Usage

```bash
# Run with default case (warehouse)
python3 src/main.py

# Run specific polygon
python3 src/main.py --case maze
python3 src/main.py --case bottleneck
python3 src/main.py --case thin_walls
python3 src/main.py --case warehouse

# With Field of View enabled
python3 src/main.py --case maze --fov 5

# Adjust simulation speed (ms per tick)
python3 src/main.py --case bottleneck --speed 150

# Combine options
python3 src/main.py --case thin_walls --fov 3 --speed 100
```

## Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume |
| `Q` or `Esc` | Quit |

## Adding a New Polygon

Place a folder with the following files into `data/`:

| File | Description |
|------|-------------|
| `map.png` | Map image |
| `scenario.json` | Agent positions and tasks |
| `edges.csv` | Navigation graph edges |
| `metrics.json` | Grid dimensions |
| `tasks.json` | Task list (MAPD only) |

Then run:
```bash
python3 src/main.py --case your_polygon_name
```

No code changes required.

## Experimental Results

| Polygon | Type | Agents | Tasks | Makespan | Collisions |
|---------|------|--------|-------|----------|------------|
| warehouse | MAPD | 5 | 10/10 | 217 | 0 |
| bottleneck | MAPF | 6 | — | 109 | 0 |
| maze | MAPF | 6 | — | 267 | 0 |
| thin_walls | MAPF | 8 | — | 86 | 0 |

## Requirements

- Python 3.8+
- OpenCV
- NumPy

## Author

Karaeva Arina Olegovna  
HSE University, Faculty of Computer Science  
Bachelor's Programme "Data Science and Business Analytics", 2nd year