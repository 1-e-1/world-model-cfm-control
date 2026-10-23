# External assets and optional tools

Isaac Sim and IsaacLab must be installed independently under their own terms.
Run simulator commands using their supported Python environment. `ISAACLAB_ROOT`
can point to the local IsaacLab installation. No simulator installation or shared
Python environment is changed by the publication process.

Original robot USD/CAD scenes, referenced meshes/materials, calibration, datasets
and trained weights are not bundled. Restore the exact robot version, joint order,
TCP/base transforms and limits required by the chosen original configuration before
attempting a simulation or new training run. Historical checkpoint and scene paths
in README/configuration files are references, not included artifacts.

Optional old asset-build scripts can use `ZBOT_V63_ROOT` to locate a separately
prepared ZBot v6.3 asset workspace. CAD reference helpers can use
`SOLIDWORKS_AUTOMATION_SCRIPTS`; they are not needed for core numerical tests.
cuRobo/Warp/FCL integrations need separately installed compatible libraries and
validated geometry. No external cuRobo tree is copied or relicensed here.
