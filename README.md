[![release](https://img.shields.io/github/release/ISISNeutronMuon/NXTomoTesting.svg)](https://github.com/ISISNeutronMuon/NXTomoTesting/releases)
[![Actions Status](https://github.com/ISISNeutronMuon/NXTomoTesting/workflows/Build/badge.svg)](https://github.com/ISISNeutronMuon/NXTomoTesting/actions)

NXtomoWriter
============
NXtomoWriter provides a function to convert tomography data stored as TIFF image files to a Nexus compliant file format 
(using [NXtomo](https://manual.nexusformat.org/classes/applications/NXtomo.html))
  
How to run the code
-------------------
The code is Python 3 compatible. To run the source: 

1. Download the repository

2. Install dependencies
        
        pip install -r requirements.txt

3. Open python and type

        import nxtomowriter as ntw

Build instruction
-----------------
To build the wheel 

        python setup.py bdist_wheel

To buid the GUI

        PyInstaller build.spec


