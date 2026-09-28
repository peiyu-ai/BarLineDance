# SMPL UV layout and appearance textures

Downloaded 2026-08-29 from https://github.com/dancasas/SMPLitex (BMVC 2023).

* `smpl_uv.obj` — the official SMPL UV unwrap (`sample-data/smpl_uv_20200910/`).
  6,890 vertices / 7,576 UV coordinates / 13,776 faces, matching
  `SMPL_*_clean.pkl` exactly.  The UV count exceeds the vertex count because the
  seams are split; a textured render therefore has to un-weld the mesh so each
  face corner carries its own coordinate, which is what
  `tools/render_avatar_video.load_uv` does.
* `dancer_female_01.png` — one SMPLitex result texture (`results/DeepFashion/
  WOMEN-Blouses_Shirts-id_00000165-02_1_front_texture_inpaint-001_...png`),
  512x512 in that UV space.

**These are appearance only.**  Nothing here changes a joint, a rotation, or a
translation: the texture is sampled on the same mesh the untextured render draws,
and the joint gate in `render_avatar_video.skin` runs either way.  A prettier
figure must not be able to make a worse generation look better, which is why the
stick-figure renderer stays and why every arm in one stack wears the same clothes.
