# Body-model assets (not included)

The code expects these; none of the body models is redistributed here because of their licences.

* `SMPL_MALE_clean.pkl`, `SMPL_FEMALE_clean.pkl`, `SMPL_NEUTRAL_clean.pkl`: the official SMPL model files
  from <https://smpl.is.tue.mpg.de/> (after accepting its licence) with their chumpy arrays converted to
  plain numpy, which is what `_clean` means.
* SMPL-X, for the 3D render: `SMPLX_<GENDER>.npz` under GVHMR's `inputs/checkpoints/body_models/smplx/`.
* `smpl_vert_segmentation.json`, `smplx_vert_segmentation.json`: vertex-to-body-part lists from Meshcapade's
  public wiki (`assets/SMPL_body_segmentation/`), used only to colour the 3D render.
* `uv/`: see `uv/PROVENANCE.md`; the CC BY 4.0 UV template is included, the texture is not.
