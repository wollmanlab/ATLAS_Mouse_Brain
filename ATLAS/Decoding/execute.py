#!/usr/bin/env python
import argparse
from ATLAS.Utils.decodeu import *

# conda activate ATLAS_3.9; nohup python -W ignore /home/zach/PythonRepos/ATLAS/ATLAS/Analysis/execute.py all -p /scratchdata1/Images2024/Zach/MouseBrainAtlas -a /scratchdata1/MouseBrainAtlases_V3 > /scratchdata1/MouseBrainAtlases_V3/execute.log 2>&1 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("animal", type=str, help="Name of the animal to analyze")
    parser.add_argument("-p", "--project_path", type=str, dest="project_path", default='/scratchdata1/Images2024/Zach/MouseBrainAtlas', action='store', help="Path to the project directory (default: '/scratchdata1/Images2024/Zach/MouseBrainAtlas')")
    parser.add_argument("-a", "--analysis_path", type=str, dest="analysis_path", default='/scratchdata1/MouseBrainAtlases_V3', action='store', help="Path to the analysis directory (default: '/scratchdata1/MouseBrainAtlases_V0')")

    args = parser.parse_args()
    if not os.path.exists(args.analysis_path):
        os.mkdir(args.analysis_path,mode=0o777)


    analyze_mouse_brain_data(args.animal,
                        project_path=args.project_path,
                        analysis_path=args.analysis_path,
                        verbose=False,repair=True)
    print(f"Analysis of {args.animal} complete")